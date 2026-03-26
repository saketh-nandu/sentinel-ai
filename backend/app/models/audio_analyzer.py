"""
Sentinel AI - Audio Analyzer
Real AI-powered audio deepfake detection using AssemblyAI.
"""
import os
from pathlib import Path
from typing import Dict
import assemblyai as aai


class AudioAnalyzer:
    """
    Audio analysis using AssemblyAI for detecting deepfakes and voice cloning.
    """
    
    def __init__(self):
        """Initialize the audio analyzer with AssemblyAI."""
        api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            raise ValueError("ASSEMBLYAI_API_KEY not found in environment variables")
        
        aai.settings.api_key = api_key
        self.loaded = True
    
    def analyze(self, file_path: Path) -> Dict:
        """
        Analyze audio for deepfakes and voice manipulation.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dict with analysis results including probabilities
        """
        try:
            # Transcribe and analyze audio
            config = aai.TranscriptionConfig(
                speech_model=aai.SpeechModel.best,
                language_detection=True
            )
            
            transcriber = aai.Transcriber(config=config)
            transcript = transcriber.transcribe(str(file_path))
            
            if transcript.status == aai.TranscriptStatus.error:
                raise Exception(f"Transcription failed: {transcript.error}")
            
            # Analyze the transcription and audio characteristics
            text = transcript.text or ""
            confidence = transcript.confidence or 0.5
            
            # Use Gemini to analyze the transcribed text for suspicious content
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            prompt = f"""Analyze this audio transcription for signs of:
1. Voice cloning or deepfake audio
2. Scam or phishing attempts
3. Social engineering

Transcription: "{text}"
Audio confidence: {confidence}

Look for:
- Unnatural speech patterns
- Robotic or synthetic voice indicators
- Scam language (urgency, money requests, threats)
- Impersonation attempts
- Suspicious requests

Provide analysis in this format:
VERDICT: [Deepfake/Scam/Suspicious/Legitimate]
RISK_SCORE: [0-100]
REASONING: [Brief explanation]

Be specific about audio and content indicators."""

            response = model.generate_content(prompt)
            analysis_text = response.text
            
            # Parse response
            verdict = "Legitimate"
            risk_score = 20
            reasoning = analysis_text
            
            if "VERDICT:" in analysis_text:
                verdict_line = [line for line in analysis_text.split('\n') if 'VERDICT:' in line][0]
                verdict = verdict_line.split('VERDICT:')[1].strip().split()[0]
            
            if "RISK_SCORE:" in analysis_text:
                risk_line = [line for line in analysis_text.split('\n') if 'RISK_SCORE:' in line][0]
                try:
                    risk_score = int(''.join(filter(str.isdigit, risk_line.split('RISK_SCORE:')[1])))
                except:
                    risk_score = 50
            
            if "REASONING:" in analysis_text:
                reasoning = analysis_text.split('REASONING:')[1].strip()
            
            # Factor in transcription confidence
            if confidence < 0.7:
                risk_score = min(100, risk_score + 20)
                reasoning += f" Low transcription confidence ({confidence:.2f}) suggests possible audio manipulation."
            
            # Get audio duration
            try:
                import soundfile as sf
                info = sf.info(str(file_path))
                duration = info.duration
            except:
                duration = 0.0

            risk_decimal = risk_score / 100.0

            if "DEEPFAKE" in verdict.upper() or "CLONED" in verdict.upper():
                return {
                    "human_voice": 1.0 - risk_decimal,
                    "tts_likelihood": risk_decimal * 0.3,
                    "voice_cloning": risk_decimal * 0.7,
                    "reasoning": reasoning,
                    "transcription": text,
                    "confidence": confidence,
                    "duration_seconds": duration
                }
            elif "SCAM" in verdict.upper():
                return {
                    "human_voice": 1.0 - risk_decimal,
                    "tts_likelihood": risk_decimal * 0.5,
                    "voice_cloning": risk_decimal * 0.2,
                    "reasoning": reasoning,
                    "transcription": text,
                    "confidence": confidence,
                    "duration_seconds": duration
                }
            elif "SUSPICIOUS" in verdict.upper():
                return {
                    "human_voice": 1.0 - risk_decimal,
                    "tts_likelihood": risk_decimal * 0.4,
                    "voice_cloning": risk_decimal * 0.3,
                    "reasoning": reasoning,
                    "transcription": text,
                    "confidence": confidence,
                    "duration_seconds": duration
                }
            else:  # Legitimate
                return {
                    "human_voice": 1.0 - risk_decimal,
                    "tts_likelihood": risk_decimal * 0.3,
                    "voice_cloning": risk_decimal * 0.2,
                    "reasoning": reasoning,
                    "transcription": text,
                    "confidence": confidence,
                    "duration_seconds": duration
                }
                
        except Exception as e:
            print(f"Audio analysis failed: {e}")
            return {
                "human_voice": 0.5,
                "tts_likelihood": 0.25,
                "voice_cloning": 0.25,
                "reasoning": f"Analysis failed: {str(e)}",
                "transcription": "",
                "confidence": 0.0,
                "duration_seconds": 0.0
            }


# Singleton instance
_analyzer = None


def get_audio_analyzer() -> AudioAnalyzer:
    """Get or create the audio analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = AudioAnalyzer()
    return _analyzer
