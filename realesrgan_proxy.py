import subprocess
import os

REAL_ESRGAN_PATH = './Real-ESRGAN/'
REAL_ESRGAN_inference_PATH = 'inference_realesrgan_video.py'

def proxy_call_realesrgan(
    input_path,
    output_path='results',
    model_name='realesr-animevideov3',
    outscale=4,
    denoise_strength=0.5,
    face_enhance=False,
    tile=0,
    fps=None,
    ffmpeg_bin='ffmpeg'
):
    """
    Proxy function to call the original Real-ESRGAN script with arguments.

    Args:
        input_path (str): Path to input file or folder.
        output_path (str): Path to output folder.
        model_name (str): The model name to use.
        outscale (float): Upscaling factor.
        denoise_strength (float): Denoise strength parameter.
        face_enhance (bool): Whether to enhance faces.
        tile (int): Tile size for processing.
        fps (float, optional): Frames per second for output videos.
        ffmpeg_bin (str): Path to ffmpeg binary.
    """
    if not os.path.exists(REAL_ESRGAN_PATH): return
    
    # Construct the command to call the original script
    command = [
        "python", REAL_ESRGAN_inference_PATH,  # Replace with the actual script name
        "-i", os.path.abspath(input_path),
        "-o", os.path.abspath(output_path),
        "-n", model_name,
        "-s", str(outscale),
        "--denoise_strength", str(denoise_strength),
        "--ffmpeg_bin", ffmpeg_bin
    ]

    # Add optional arguments only if they're provided
    if face_enhance:
        command.append("--face_enhance")
    if tile > 0:
        command.extend(["-t", str(tile)])
    if fps is not None:
        command.extend(["--fps", str(fps)])

    # Print command for debugging (optional)
    print("Running command:", " ".join(command))
    
    # Set the environment variable for PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = REAL_ESRGAN_PATH

    # Run the subprocess
    subprocess.run(command, check=True, cwd=REAL_ESRGAN_PATH, env=env)
