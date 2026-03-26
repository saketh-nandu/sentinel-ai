"""
Sentinel AI - Video Analyzer
Real AI-powered video deepfake detection using Google Gemini Vision API.
"""
import os
from pathlib import Path
from typing import Dict
import google.generativeai as genai
import cv2
import tempfile


class VideoAnalyzer:
    """
    Video analysis using Google Gemini Vision API for detecting deepfakes
    and manipulated videos.
    """
    
    def __init__(self):
        """Initialize the video analyzer with Gemini API."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.loaded = True
    
    def _extract_frames(self, video_path: Path, num_frames: int = 5):
        """Extract key frames from video for analysis."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                return []
            
            # Extract frames at regular intervals
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            frames = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
            
            cap.release()
            return frames
            
        except Exception as e:
            print(f"Frame extraction failed: {e}")
            return []
    
    def analyze(self, file_path: Path) -> Dict:
        """
        Analyze video for deepfakes and manipulation.
        
        Args:
            file_path: Path to video file
            
        Returns:
            Dict with analysis results including probabilities
        """
        try:
            from PIL import Image
            import numpy as np
            
            # Extract key frames
            frames = self._extract_frames(file_path, num_frames=1)
            
            if not frames:
                raise Exception("Could not extract frames from video")
            
            # Analyze frames with Gemini
            prompt = """Analyze these video frames for signs of deepfake or manipulation.

Look for:
- Face swapping or morphing
- Unnatural facial movements or expressions
- Inconsistent lighting across frames
- Artifacts around face/body edges
- Unnatural eye movements or blinking
- Lip sync issues
- Background inconsistencies
- Digital artifacts or glitches
- Temporal inconsistencies between frames

Provide analysis in this format:
VERDICT: [Deepfake/Manipulated/Real]
CONFIDENCE: [0-100]
REASONING: [Specific observations about the video]

Be thorough and mention specific frame issues if found."""

            # Convert frames to PIL Images
            pil_frames = [Image.fromarray(frame) for frame in frames]
            
            # Send frames to Gemini
            content = [prompt] + pil_frames
            response = self.model.generate_content(content)
            analysis_text = response.text
            
            # Parse response
            verdict = "Real"
            confidence = 50
            reasoning = analysis_text
            
            if "VERDICT:" in analysis_text:
                verdict_line = [line for line in analysis_text.split('\n') if 'VERDICT:' in line][0]
                verdict = verdict_line.split('VERDICT:')[1].strip().split()[0]
            
            if "CONFIDENCE:" in analysis_text:
                conf_line = [line for line in analysis_text.split('\n') if 'CONFIDENCE:' in line][0]
                try:
                    confidence = int(''.join(filter(str.isdigit, conf_line.split('CONFIDENCE:')[1])))
                except:
                    confidence = 50
            
            if "REASONING:" in analysis_text:
                reasoning = analysis_text.split('REASONING:')[1].strip()
            
            confidence_decimal = confidence / 100.0
            
            # Get video duration
            cap_info = cv2.VideoCapture(str(file_path))
            fps = cap_info.get(cv2.CAP_PROP_FPS) or 1
            total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            cap_info.release()

            if "DEEPFAKE" in verdict.upper():
                return {
                    "real_probability": 1.0 - confidence_decimal,
                    "deepfake_likelihood": confidence_decimal * 0.9,
                    "manipulated": confidence_decimal * 0.1,
                    "reasoning": reasoning,
                    "frames_analyzed": len(frames),
                    "duration_seconds": duration
                }
            elif "MANIPULATED" in verdict.upper() or "EDITED" in verdict.upper():
                return {
                    "real_probability": 1.0 - confidence_decimal,
                    "deepfake_likelihood": confidence_decimal * 0.4,
                    "manipulated": confidence_decimal * 0.6,
                    "reasoning": reasoning,
                    "frames_analyzed": len(frames),
                    "duration_seconds": duration
                }
            else:  # Real
                return {
                    "real_probability": confidence_decimal,
                    "deepfake_likelihood": (1.0 - confidence_decimal) * 0.5,
                    "manipulated": (1.0 - confidence_decimal) * 0.5,
                    "reasoning": reasoning,
                    "frames_analyzed": len(frames),
                    "duration_seconds": duration
                }
                
        except Exception as e:
            print(f"Video analysis failed: {e}")
            return {
                "real_probability": 0.5,
                "deepfake_likelihood": 0.25,
                "manipulated": 0.25,
                "reasoning": f"Analysis failed: {str(e)}",
                "frames_analyzed": 0,
                "duration_seconds": 0.0
            }


# Singleton instance
_analyzer = None


def get_video_analyzer() -> VideoAnalyzer:
    """Get or create the video analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = VideoAnalyzer()
    return _analyzer
