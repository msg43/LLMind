"""
Audio Manager - Local Speech-to-Text and Text-to-Speech
Optimized for Apple Silicon with local Whisper and macOS TTS
"""

import asyncio
import base64
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import speech_recognition as sr
    import whisper
    from pydub import AudioSegment

    AUDIO_AVAILABLE = True
except ImportError as e:
    AUDIO_AVAILABLE = False
    print(f"⚠️  Audio libraries not available: {e}")

from config import settings


class AudioManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AudioManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if AudioManager._initialized:
            return

        self.whisper_model = None
        self.recognizer = None
        # Premium voice selection – 9 high-quality voices
        self.available_voices = [
            {
                "name": "Samantha",
                "language": "en_US",
                "description": "Premium female voice – Natural and expressive",
                "quality": "premium",
                "available": True,
            },
            {
                "name": "Daniel",
                "language": "en_GB",
                "description": "Premium British male voice – Professional and clear",
                "quality": "premium",
                "available": True,
            },
            {
                "name": "Karen",
                "language": "en_AU",
                "description": "Premium Australian female voice – Friendly and natural",
                "quality": "premium",
                "available": True,
            },
            {
                "name": "Moira",
                "language": "en_IE",
                "description": "Premium Irish female voice – Warm and engaging",
                "quality": "premium",
                "available": True,
            },
            {
                "name": "Tessa",
                "language": "en_ZA",
                "description": "Premium South-African female voice – Clear articulation",
                "quality": "premium",
                "available": True,
            },
            {
                "name": "Ralph",
                "language": "en_US",
                "description": "Professional American male voice – Deep and authoritative",
                "quality": "enhanced",
                "available": True,
            },
            {
                "name": "Fred",
                "language": "en_US",
                "description": "Classic American male voice – Reliable and clear",
                "quality": "enhanced",
                "available": True,
            },
            {
                "name": "Albert",
                "language": "en_US",
                "description": "Professional American male voice – Smooth delivery",
                "quality": "enhanced",
                "available": True,
            },
            {
                "name": "Kathy",
                "language": "en_US",
                "description": "Professional American female voice – Pleasant tone",
                "quality": "enhanced",
                "available": True,
            },
        ]
        print(f"🔊 Loaded {len(self.available_voices)} premium TTS voices")
        self.current_voice = settings.tts_voice

        # Mark as initialized
        AudioManager._initialized = True

    async def initialize(self):
        """Initialize audio components"""
        if not AUDIO_AVAILABLE:
            print(
                "⚠️  Audio features disabled - install whisper and speech_recognition"
            )
            return

        print("🎤 Initializing Audio Manager...")

        # Initialize Whisper for STT (lazy loading for faster startup)
        try:
            print("📥 Loading Whisper model (optimized for Apple Silicon)...")
            # Use tiny model for fastest loading, upgrade to base on demand
            model_size = getattr(settings, "whisper_model_size", "tiny")
            self.whisper_model = whisper.load_model(
                model_size, download_root="models/whisper"
            )
            print(f"✅ Whisper {model_size} model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading Whisper: {e}")
            self.whisper_model = None

        # Initialize speech recognition
        try:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            print("✅ Speech recognizer initialized!")
        except Exception as e:
            print(f"❌ Error initializing speech recognizer: {e}")

        # Get available TTS voices on macOS
        await self._get_available_voices()

        print("✅ Audio Manager initialized successfully!")

    async def speech_to_text(self, audio_data: str) -> str:
        """Convert speech to text using local Whisper"""
        try:
            if not self.whisper_model:
                return "Speech recognition not available"

            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name

            try:
                # Run Whisper transcription in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self.whisper_model.transcribe, temp_path
                )

                text = result["text"].strip()
                print(f"🎤 Transcribed: {text[:100]}...")
                return text

            finally:
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)

        except Exception as e:
            print(f"❌ Error in speech-to-text: {e}")
            return f"Error: {str(e)}"

    async def text_to_speech(self, text: str, voice: str = None) -> str:
        """Convert text to speech using macOS built-in TTS and return base64 WAV."""
        try:
            voice = voice or self.current_voice or "Samantha"
            # Create temporary output file (AIFF for generation, then convert to WAV)
            with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp_out:
                aiff_path = tmp_out.name

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                wav_path = tmp_wav.name

            # Generate speech to AIFF file
            process = await asyncio.create_subprocess_exec(
                "say",
                "-v",
                voice,
                "-r",
                str(settings.tts_rate),
                "-o",
                aiff_path,
                text,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            if process.returncode != 0:
                raise Exception(stderr.decode())

            # Convert AIFF to WAV using afconvert
            convert_process = await asyncio.create_subprocess_exec(
                "afconvert",
                aiff_path,
                wav_path,
                "-d",
                "LEI16@44100",  # Linear PCM 16-bit @ 44.1kHz
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, convert_stderr = await convert_process.communicate()
            if convert_process.returncode != 0:
                raise Exception(f"Audio conversion failed: {convert_stderr.decode()}")

            # Read the WAV file and base64-encode
            with open(wav_path, "rb") as f:
                audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode()

            # Clean up temp files
            Path(aiff_path).unlink(missing_ok=True)
            Path(wav_path).unlink(missing_ok=True)

            print(f"🔊 Generated TTS ({voice}) for: {text[:40]}… [WAV format]")
            return audio_b64
        except Exception as e:
            print(f"❌ Error in text-to-speech: {e}")
            return ""

    async def text_to_speech_file(
        self, text: str, output_path: str, voice: str = None
    ) -> bool:
        """Convert text to speech and save as file with enhanced quality"""
        try:
            voice = voice or self.current_voice

            # Use macOS say command with enhanced quality settings
            process = await asyncio.create_subprocess_exec(
                "say",
                "-v",
                voice,
                "-r",
                str(settings.tts_rate),
                "-q",
                "1",  # High quality flag
                "--file-format=WAVE",  # Ensure WAVE format for best quality
                "--data-format=LEF32@44100",  # 32-bit float at 44.1kHz
                "-o",
                output_path,
                text,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                print(f"🔊 TTS saved to: {output_path}")
                return True
            else:
                print(f"❌ TTS failed: {stderr.decode()}")
                return False

        except Exception as e:
            print(f"❌ Error generating TTS file: {e}")
            return False

    async def _get_available_voices(self):
        """Populate curated premium voices; skip system enumeration to avoid parsing errors"""
        self.available_voices = [
            {
                "name": "Samantha",
                "language": "en_US",
                "description": "Premium female voice – Natural and expressive",
                "quality": "premium",
                "available": True,
            },
            {
                "name": "Daniel",
                "language": "en_GB",
                "description": "Premium British male voice – Professional and clear",
                "quality": "premium",
                "available": True,
            },
            {
                "name": "Karen",
                "language": "en_AU",
                "description": "Premium Australian female voice – Friendly and natural",
                "quality": "premium",
                "available": True,
            },
            {
                "name": "Moira",
                "language": "en_IE",
                "description": "Premium Irish female voice – Warm and engaging",
                "quality": "premium",
                "available": True,
            },
            {
                "name": "Tessa",
                "language": "en_ZA",
                "description": "Premium South-African female voice – Clear articulation",
                "quality": "premium",
                "available": True,
            },
            {
                "name": "Ralph",
                "language": "en_US",
                "description": "Professional American male voice – Deep and authoritative",
                "quality": "enhanced",
                "available": True,
            },
            {
                "name": "Fred",
                "language": "en_US",
                "description": "Classic American male voice – Reliable and clear",
                "quality": "enhanced",
                "available": True,
            },
            {
                "name": "Albert",
                "language": "en_US",
                "description": "Professional American male voice – Smooth delivery",
                "quality": "enhanced",
                "available": True,
            },
            {
                "name": "Kathy",
                "language": "en_US",
                "description": "Professional American female voice – Pleasant tone",
                "quality": "enhanced",
                "available": True,
            },
        ]
        print(f"🔊 Loaded {len(self.available_voices)} premium TTS voices")

    async def get_voices(self) -> list:
        """Get available TTS voices - return premium curated list"""
        if not self.available_voices:
            await self._get_available_voices()
        return self.available_voices

    async def set_voice(self, voice_name: str) -> bool:
        """Set current TTS voice"""
        try:
            available_names = [v["name"] for v in self.available_voices]
            if voice_name in available_names:
                self.current_voice = voice_name
                print(f"🔊 Voice changed to: {voice_name}")
                return True
            else:
                print(f"⚠️  Voice not found: {voice_name}")
                return False
        except Exception as e:
            print(f"❌ Error setting voice: {e}")
            return False

    async def test_voice(self, voice_name: str = None) -> bool:
        """Test a TTS voice"""
        try:
            voice = voice_name or self.current_voice
            test_text = "Hello! This is a test of the LLMind voice system."

            process = await asyncio.create_subprocess_exec(
                "say",
                "-v",
                voice,
                test_text,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()
            return process.returncode == 0

        except Exception as e:
            print(f"❌ Error testing voice: {e}")
            return False

    async def record_audio(self, duration: int = 5) -> Optional[str]:
        """Record audio from microphone (placeholder - would need platform-specific implementation)"""
        try:
            if not self.recognizer:
                return None

            # This is a simplified version - in practice, you'd want WebRTC or similar
            # for real-time recording from the web interface
            print(f"🎤 Recording audio for {duration} seconds...")

            # For now, return a placeholder
            # Real implementation would integrate with web browser's MediaRecorder API
            return "audio_recording_placeholder"

        except Exception as e:
            print(f"❌ Error recording audio: {e}")
            return None

    def get_audio_settings(self) -> Dict[str, Any]:
        """Get current audio settings with quality recommendations"""
        return {
            "current_voice": self.current_voice,
            "tts_rate": settings.tts_rate,
            "stt_language": settings.stt_language,
            "audio_sample_rate": settings.audio_sample_rate,
            "whisper_available": self.whisper_model is not None,
            "voices_count": len(self.available_voices),
            "quality_recommendations": {
                "best_voices": ["Victoria", "Daniel", "Samantha"],
                "optimal_rate_range": "160-200",
                "current_quality": (
                    "High" if settings.audio_sample_rate >= 44100 else "Standard"
                ),
                "recommendations": [
                    "Use Victoria for professional female voice",
                    "Use Daniel for professional British male voice",
                    "Set rate between 160-200 for natural speech",
                    "Samantha provides premium quality as alternative",
                ],
            },
        }

    async def optimize_voice_settings(
        self, voice_preference: str = "female"
    ) -> Dict[str, str]:
        """Get optimized voice settings based on user preference"""
        recommendations = {
            "female": {
                "voice": "Victoria",
                "rate": "165",
                "description": "Professional, clear female voice",
            },
            "male": {
                "voice": "Daniel",
                "rate": "170",
                "description": "Professional British male voice",
            },
            "british": {
                "voice": "Daniel",
                "rate": "170",
                "description": "Professional British accent",
            },
            "professional": {
                "voice": "Victoria",
                "rate": "165",
                "description": "Premium professional female voice",
            },
        }

        return recommendations.get(voice_preference, recommendations["female"])

    async def update_settings(self, new_settings: Dict[str, Any]):
        """Update audio settings"""
        try:
            if "voice" in new_settings:
                await self.set_voice(new_settings["voice"])

            if "tts_rate" in new_settings:
                settings.tts_rate = max(50, min(500, int(new_settings["tts_rate"])))

            if "stt_language" in new_settings:
                settings.stt_language = new_settings["stt_language"]

            print("🔧 Audio settings updated")

        except Exception as e:
            print(f"❌ Error updating audio settings: {e}")

    async def get_system_audio_info(self) -> Dict[str, Any]:
        """Get system audio information"""
        try:
            # Check if we're on macOS
            process = await asyncio.create_subprocess_exec(
                "uname",
                "-s",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            system = stdout.decode().strip()

            return {
                "system": system,
                "macos_tts": system == "Darwin",
                "whisper_model": "base" if self.whisper_model else None,
                "audio_libraries": AUDIO_AVAILABLE,
            }

        except Exception as e:
            return {
                "system": "unknown",
                "macos_tts": False,
                "whisper_model": None,
                "audio_libraries": AUDIO_AVAILABLE,
                "error": str(e),
            }
