import requests
import time
import base64
import argparse
import os
import json
from io import BytesIO
from PIL import Image
import urllib3

# Disable SSL warnings (for testing only)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="Test client for RunPod FLUX.1-dev image generation API")
    parser.add_argument("--endpoint", type=str, required=True, help="RunPod endpoint ID")
    parser.add_argument("--api_key", type=str, required=True, help="RunPod API key")
    parser.add_argument("--prompt", type=str, default="A beautiful mountain landscape at sunset", help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality", help="Negative prompt")
    parser.add_argument("--height", type=int, default=768, help="Image height (default: 768)")
    parser.add_argument("--width", type=int, default=768, help="Image width (default: 768)")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps (default: 30)")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale (default: 7.5)")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate (default: 1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: None)")
    parser.add_argument("--output", type=str, default="generated_image.png", help="Output image path (default: generated_image.png)")
    return parser.parse_args()

def submit_job(api_key, endpoint_id, job_input):
    """Submit a job to the RunPod serverless API"""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    # Disable SSL verification for testing purposes
    response = requests.post(url, headers=headers, json={"input": job_input}, verify=False)
    return response.json()

def check_status(api_key, endpoint_id, job_id):
    """Check the status of a job"""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    # Disable SSL verification for testing purposes
    response = requests.get(url, headers=headers, verify=False)
    return response.json()

def main():
    args = parse_args()
    
    # Prepare job input
    job_input = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "num_images": args.num_images,
    }
    
    if args.seed is not None:
        job_input["seed"] = args.seed
    
    # Submit job
    print(f"Submitting job to endpoint {args.endpoint}...")
    response = submit_job(args.api_key, args.endpoint, job_input)
    
    if "id" not in response:
        print(f"Error submitting job: {response}")
        return
    
    job_id = response["id"]
    print(f"Job submitted successfully. Job ID: {job_id}")
    
    # Poll for job completion
    print("Waiting for job to complete...")
    print("NOTE: First run can take 5-10 minutes for container startup")
    start_time = time.time()
    
    while True:
        status_data = check_status(args.api_key, args.endpoint, job_id)
        status = status_data.get("status")
        
        if status == "COMPLETED":
            elapsed_time = time.time() - start_time
            print(f"Job completed in {elapsed_time:.2f} seconds")
            
            output = status_data.get("output", {})
            
            if "error" in output:
                print(f"Error in job: {output['error']}")
                return
            
            if args.num_images == 1:
                if "image" in output:
                    # Decode and save the image
                    img_data = base64.b64decode(output["image"])
                    img = Image.open(BytesIO(img_data))
                    img.save(args.output)
                    print(f"Image saved to {args.output}")
                    
                    if "metrics" in output:
                        print(f"Generation time: {output['metrics'].get('generation_time', 'unknown')} seconds")
                else:
                    print("No image data in response")
            else:
                if "images" in output:
                    # Save multiple images
                    for i, img_data in enumerate(output["images"]):
                        output_path = f"{os.path.splitext(args.output)[0]}_{i}{os.path.splitext(args.output)[1]}"
                        img = Image.open(BytesIO(base64.b64decode(img_data)))
                        img.save(output_path)
                        print(f"Image {i+1} saved to {output_path}")
                    
                    if "metrics" in output:
                        print(f"Generation time: {output['metrics'].get('generation_time', 'unknown')} seconds")
                else:
                    print("No images data in response")
            
            # Print any message if available (for development/demo mode)
            if "message" in output:
                print(f"Message: {output['message']}")
                if "input_params" in output:
                    print("Input parameters used:")
                    print(json.dumps(output["input_params"], indent=2))
            
            break
        elif status == "FAILED":
            print(f"Job failed: {status_data}")
            break
        
        # Print status and continue polling
        elapsed = time.time() - start_time
        if status:
            print(f"Current status: {status} - Elapsed time: {elapsed:.1f}s")
        else:
            print(f"Waiting for job to start... - Elapsed time: {elapsed:.1f}s")
        
        time.sleep(2)  # Poll every 2 seconds

if __name__ == "__main__":
    main()