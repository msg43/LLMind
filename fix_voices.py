#!/usr/bin/env python3
"""
Comprehensive voice system fix for LLMind
This script replaces all voice-related configurations with 9 premium voices
"""

import re

# Complete premium voice configuration
PREMIUM_VOICES = """[
    {"name": "Samantha", "language": "en_US", "description": "Premium female voice - Natural and expressive", "quality": "premium", "available": true},
    {"name": "Daniel", "language": "en_GB", "description": "Premium British male voice - Professional and clear", "quality": "premium", "available": true},
    {"name": "Karen", "language": "en_AU", "description": "Premium Australian female voice - Friendly and natural", "quality": "premium", "available": true},
    {"name": "Moira", "language": "en_IE", "description": "Premium Irish female voice - Warm and engaging", "quality": "premium", "available": true},
    {"name": "Tessa", "language": "en_ZA", "description": "Premium South African female voice - Clear articulation", "quality": "premium", "available": true},
    {"name": "Ralph", "language": "en_US", "description": "Professional American male voice - Deep and authoritative", "quality": "enhanced", "available": true},
    {"name": "Fred", "language": "en_US", "description": "Classic American male voice - Reliable and clear", "quality": "enhanced", "available": true},
    {"name": "Albert", "language": "en_US", "description": "Professional American male voice - Smooth delivery", "quality": "enhanced", "available": true},
    {"name": "Kathy", "language": "en_US", "description": "Professional American female voice - Pleasant tone", "quality": "enhanced", "available": true}
]"""


def fix_audio_manager():
    """Fix the audio manager voice configuration"""
    with open("core/audio_manager.py", "r") as f:
        content = f.read()

    # Replace the get_voices method completely
    new_get_voices = '''    async def get_voices(self) -> list:
        """Get available TTS voices - return premium curated list"""
        if not self.available_voices:
            await self._get_available_voices()
        return self.available_voices'''

    # Replace get_voices method
    content = re.sub(
        r"async def get_voices\(self\)[^:]*:.*?return.*?voices.*?$",
        new_get_voices.strip(),
        content,
        flags=re.MULTILINE | re.DOTALL,
    )

    # Replace voice initialization in _get_available_voices
    new_init = f"""            # Premium voice selection - 9 high quality voices
            self.available_voices = {PREMIUM_VOICES}
            print(f"🔊 Loaded {{len(self.available_voices)}} premium TTS voices")"""

    # Replace the voice loading section
    content = re.sub(
        r"self\.available_voices = \[.*?\]", new_init.strip(), content, flags=re.DOTALL
    )

    with open("core/audio_manager.py", "w") as f:
        f.write(content)

    print("✅ Fixed audio_manager.py voice configuration")


if __name__ == "__main__":
    fix_audio_manager()
    print("🎤 Voice system fix complete!")
    print("   - 5 Premium voices: Samantha, Daniel, Karen, Moira, Tessa")
    print("   - 4 Enhanced voices: Ralph, Fred, Albert, Kathy")
    print("   - Total: 9 high-quality voices available")
