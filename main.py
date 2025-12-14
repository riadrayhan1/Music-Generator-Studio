import os
import torch
import scipy.io.wavfile
import scipy.signal
import gradio as gr
import numpy as np
import warnings
import traceback
warnings.filterwarnings('ignore')

# Auto-detect GPU/CPU with GPU priority
if torch.cuda.is_available():
    device = "cuda"
    print(f"🎮 GPU DETECTED: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("💻 Using CPU mode")
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

class ProMusicGenerator:
    def __init__(self):
        self.music_model = None
        self.processor = None
        self.vocoder = None
        self.tts_model = None
        self.tts_processor = None
        self.speaker_embeddings = None
        self.model_ready = False
        self.sample_rate = 32000
        self.error_log = []
        
    def create_default_speaker_embeddings(self):
        torch.manual_seed(42)
        female_embedding = torch.randn(1, 512) * 0.1
        torch.manual_seed(123)
        male_embedding = torch.randn(1, 512) * 0.1
        male_embedding[:, :100] *= 1.5
        return {"female": female_embedding, "male": male_embedding}
        
    def load_models(self):
        try:
            print(f"\n{'='*60}\nINITIALIZING AI MUSIC SYSTEM ({device.upper()})\n{'='*60}")
            
            try:
                from transformers import (
                    AutoProcessor, MusicgenForConditionalGeneration,
                    SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
                )
            except ImportError as e:
                error_msg = f"Missing package: {e}\n\nInstall: pip install transformers datasets soundfile scipy torch gradio sentencepiece"
                self.error_log.append(error_msg)
                return False
            
            print("\n[1/3] Loading Music Generation Model...")
            try:
                self.processor = AutoProcessor.from_pretrained(
                    "facebook/musicgen-stereo-small", cache_dir="./model_cache"
                )
                dtype = torch.float16 if device == "cuda" else torch.float32
                self.music_model = MusicgenForConditionalGeneration.from_pretrained(
                    "facebook/musicgen-stereo-small",
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    cache_dir="./model_cache"
                )
                self.music_model.to(device)
                self.music_model.eval()
                print("      ✓ Music model loaded")
            except Exception as e:
                self.error_log.append(f"Music model error: {e}")
                traceback.print_exc()
                return False
            
            print("\n[2/3] Loading Voice Synthesis...")
            try:
                self.tts_processor = SpeechT5Processor.from_pretrained(
                    "microsoft/speecht5_tts", cache_dir="./model_cache"
                )
                self.tts_model = SpeechT5ForTextToSpeech.from_pretrained(
                    "microsoft/speecht5_tts", cache_dir="./model_cache"
                )
                self.tts_model.to(device)
                self.tts_model.eval()
                self.vocoder = SpeechT5HifiGan.from_pretrained(
                    "microsoft/speecht5_hifigan", cache_dir="./model_cache"
                )
                self.vocoder.to(device)
                self.vocoder.eval()
                print("      ✓ Voice synthesis loaded")
            except Exception as e:
                self.error_log.append(f"Voice synthesis disabled: {e}")
            
            print("\n[3/3] Loading Voice Embeddings...")
            self.speaker_embeddings = self.create_default_speaker_embeddings()
            
            self.model_ready = True
            print(f"\n{'='*60}\n✅ SYSTEM READY\n{'='*60}\n")
            return True
            
        except Exception as e:
            self.error_log.append(f"CRITICAL ERROR: {e}")
            traceback.print_exc()
            return False
    
    def generate_complete_track(self, genre, custom_desc, lyrics, voice_type, 
                               duration, tempo, mood, enable_vocals):
        if not self.model_ready:
            return None, f"❌ System not initialized\n\n" + "\n".join(self.error_log)
        
        try:
            prompt_parts = []
            if genre != "custom":
                prompt_parts.append(genre)
            if custom_desc:
                prompt_parts.append(custom_desc)
            prompt_parts.extend([f"{tempo} tempo", f"{mood} atmosphere", "professional production"])
            full_prompt = ", ".join(prompt_parts)
            
            print(f"\n{'='*60}\nGENERATING TRACK\n{'='*60}")
            print(f"Genre: {genre} | Tempo: {tempo} | Mood: {mood}")
            print(f"Vocals: {enable_vocals} | Duration: {duration}s")
            print(f"Device: {device.upper()}")
            
            print(f"\n[1/3] Generating instrumental...")
            music_audio = self.generate_music(full_prompt, duration)
            if music_audio is None:
                return None, "❌ Music generation failed"
            
            vocals_available = all([self.tts_model, self.vocoder, self.speaker_embeddings])
            
            if enable_vocals and lyrics and lyrics.strip() and vocals_available:
                print(f"\n[2/3] Generating vocals...")
                vocal_audio = self.generate_vocals(lyrics, voice_type)
                if vocal_audio is not None:
                    print(f"\n[3/3] Mixing & mastering...")
                    final_audio = self.professional_mix(music_audio, vocal_audio)
                    track_type = "Full Production (Vocals + Instrumental)"
                else:
                    final_audio = music_audio
                    track_type = "Instrumental Only"
            else:
                final_audio = music_audio
                track_type = "Instrumental Only"
            
            final_audio = self.apply_mastering(final_audio)
            
            output_path = "professional_track.wav"
            scipy.io.wavfile.write(output_path, rate=self.sample_rate, data=final_audio)
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            
            print(f"\n{'='*60}\n✅ COMPLETED: {track_type}\n{'='*60}\n")
            
            status_msg = f"✅ Track generated successfully!\n\n🎵 {track_type}\n📁 {file_size:.2f} MB | ⏱️ {duration}s\n🖥️ Device: {device.upper()}"
            return output_path, status_msg
            
        except Exception as e:
            error_msg = f"❌ Generation Error: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return None, error_msg
    
    def generate_music(self, prompt, duration):
        try:
            inputs = self.processor(text=[prompt], padding=True, return_tensors="pt").to(device)
            max_tokens = int(duration * 50)
            
            with torch.no_grad():
                audio_values = self.music_model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=True,
                    temperature=1.0, top_k=250, guidance_scale=3.0
                )
            
            if len(audio_values.shape) > 2 and audio_values.shape[1] == 2:
                audio = audio_values[0].mean(dim=0).cpu().numpy()
            else:
                audio = audio_values[0, 0].cpu().numpy()
            
            audio = audio / (np.max(np.abs(audio)) + 1e-9)
            return np.int16(audio * 32767)
        except Exception as e:
            print(f"Music generation error: {e}")
            traceback.print_exc()
            return None
    
    def generate_vocals(self, lyrics, voice_type):
        try:
            clean_lyrics = lyrics.replace("\n", " ").strip()[:500]
            inputs = self.tts_processor(text=clean_lyrics, return_tensors="pt").to(device)
            speaker_embedding = self.speaker_embeddings.get(voice_type, self.speaker_embeddings["female"]).to(device)
            
            with torch.no_grad():
                speech = self.tts_model.generate_speech(
                    inputs["input_ids"], speaker_embedding, vocoder=self.vocoder
                )
            
            vocal_audio = speech.cpu().numpy()
            num_samples = int(len(vocal_audio) * self.sample_rate / 16000)
            vocal_audio = scipy.signal.resample(vocal_audio, num_samples)
            vocal_audio = vocal_audio / (np.max(np.abs(vocal_audio)) + 1e-9)
            return np.int16(vocal_audio * 32767)
        except Exception as e:
            print(f"Vocal generation error: {e}")
            return None
    
    def professional_mix(self, music, vocals):
        try:
            target_len = max(len(music), len(vocals))
            music = np.pad(music, (0, target_len - len(music))) if len(music) < target_len else music[:target_len]
            vocals = np.pad(vocals, (0, target_len - len(vocals))) if len(vocals) < target_len else vocals[:target_len]
            
            music_f = music.astype(np.float32) / 32767.0
            vocals_f = vocals.astype(np.float32) / 32767.0
            mixed = (vocals_f * 0.75) + (music_f * 0.35)
            
            threshold, ratio = 0.7, 3.0
            mixed = np.where(np.abs(mixed) > threshold,
                           np.sign(mixed) * (threshold + (np.abs(mixed) - threshold) / ratio),
                           mixed)
            
            mixed = mixed / (np.max(np.abs(mixed)) + 1e-9)
            return np.int16(mixed * 32767)
        except Exception as e:
            print(f"Mixing error: {e}")
            return music
    
    def apply_mastering(self, audio):
        try:
            audio_f = audio.astype(np.float32) / 32767.0
            audio_f = np.clip(audio_f * 0.98, -0.98, 0.98)
            return np.int16(audio_f * 32767)
        except:
            return audio


generator = ProMusicGenerator()
system_ready = generator.load_models()

def generate_ui(genre, custom_desc, lyrics, voice_type, duration, tempo, mood, enable_vocals):
    if not system_ready:
        return None, f"❌ System not ready\n\n" + "\n".join(generator.error_log)
    if genre == "custom" and not custom_desc:
        return None, "❌ Provide custom description for 'custom' genre"
    if enable_vocals and not lyrics:
        return None, "❌ Provide lyrics for vocal generation"
    duration = min(max(duration, 10), 30)
    return generator.generate_complete_track(genre, custom_desc, lyrics, voice_type, duration, tempo, mood, enable_vocals)


app = gr.Blocks(title="Riad Music Generator")

with app:
    
    gr.HTML("<div style='text-align: center;'><h1 style='color: #667eea; font-size: 2.5em;'>🎵 Riad Music Generator Studio</h1><p style='color: #666; font-size: 1.1em;'>UNLIMITED MUSIC GENERATOR STUDIO</p></div>")
    
    with gr.Row():
        with gr.Column(scale=1):
            genre_select = gr.Dropdown(
                choices=["pop music", "rock music", "electronic dance music (EDM)", "jazz music", 
                        "classical orchestral", "hip hop beat", "lo-fi chill hop", "ambient atmospheric", 
                        "acoustic folk", "custom"],
                value="electronic dance music (EDM)", label="🎸 Genre"
            )
            
            custom_input = gr.Textbox(label="🎨 Custom Description", placeholder="e.g., heavy bass, energetic synths", lines=2)
            
            with gr.Row():
                tempo_select = gr.Dropdown(choices=["slow", "moderate", "fast"], value="moderate", label="⏱️ Tempo")
                mood_select = gr.Dropdown(
                    choices=["energetic", "calm and relaxing", "happy and upbeat", "dark and mysterious", "emotional"],
                    value="energetic", label="🎭 Mood"
                )
            
            duration_slider = gr.Slider(10, 30, value=15, step=5, label="⏲️ Duration (seconds)")
            
            gr.Markdown("### 🎤 Vocal Settings")
            enable_vocals_checkbox = gr.Checkbox(label="Enable Vocals", value=False)
            voice_type_radio = gr.Radio(choices=["female", "male"], value="female", label="🎙️ Voice")
            lyrics_input = gr.Textbox(label="📝 Lyrics", placeholder="Enter lyrics...", lines=4)
            
            generate_btn = gr.Button("🎵 Generate Track", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            audio_player = gr.Audio(label="🎧 Generated Track", type="filepath")
            status_display = gr.Textbox(label="📊 Status", lines=8, interactive=False)
            
            gr.Markdown(f"""
            ### ℹ️ System Info
            **Device:** {device.upper()} {'🎮 GPU Acceleration' if device == 'cuda' else '💻 CPU Mode'}  
            **Generation Time:** 5-8 min (instrumental) | 8-12 min (with vocals)  
            **Tip:** Start without vocals for faster testing
            """)
    
    generate_btn.click(
        fn=generate_ui,
        inputs=[genre_select, custom_input, lyrics_input, voice_type_radio, 
                duration_slider, tempo_select, mood_select, enable_vocals_checkbox],
        outputs=[audio_player, status_display]
    )

if __name__ == "__main__":
    print(f"\n{'='*60}\n🚀 LAUNCHING WEB INTERFACE\n{'='*60}\n")
    app.launch(share=False, show_error=True, inbrowser=True)