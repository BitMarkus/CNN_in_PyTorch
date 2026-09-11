# Diffusion-Based Phenotypic Extrapolation
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: MIT

# ===== Standard Library Imports =====
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
# ===== Third-Party Imports =====
import requests


#############################################################################################################
# CONFIGURATION
#
# Edit the values in this block to match your local setup.
# Nothing else in this file needs to be changed for a normal run.

# ===== Workflow selection =====
# Options:
#   "double_lora": Loads the LoRA twice (nodes 5 and 6 in the workflow).
#                  This is the workflow that generated all synthetic images
#                  reported in the paper. Use this to reproduce published results.
#   "single_lora": Loads the LoRA once (node 5 in the workflow).
#                  Cleaner and recommended for new work.
WORKFLOW_TYPE = "double_lora"

# ===== Paths =====
# Folder containing the LoRA checkpoint files (.safetensors)
CHECKPOINTS_DIR = Path("<path/to/lora/checkpoints>")

# ComfyUI output folder (where generated images are saved)
OUTPUT_DIR = Path("<path/to/comfyui/output>")

# ===== ComfyUI server =====
COMFYUI_SERVER_URL = "http://127.0.0.1:8188"

# ===== Checkpoint naming =====
# Options:
#   "kohya":   Kohya_SS format (*-XXXXXX.safetensors, 6-digit step numbers)
#   "comfyui": ComfyUI format (*-stepXXXXX.safetensors)
CHECKPOINT_FORMAT = "kohya"

# ===== Generation parameters =====
NUM_IMAGES = 10               # Number of frames per morphing series
NUM_SERIES = 1000             # Target number of series per checkpoint

# ===== Morph range =====
# 1.0 = pure WT, 0.0 = pure KO. Values beyond this range extrapolate.
MORPH_START = 1.0
MORPH_END = 0.0

# ===== Seed mode =====
# "fixed_per_series": one seed per series (all frames share the same seed)
# "random_per_image": a different random seed for each frame in a series
SEED_MODE = "fixed_per_series"

# ===== LoRA strength =====
LORA_STRENGTH_MODEL = 1.0
LORA_STRENGTH_CLIP = 1.0

#############################################################################################################


class FibroblastMorphGenerator:

    #############################################################################################################
    # CONSTRUCTOR

    # Generate morphing series by interpolating between wild-type (WT) and knockout (KO)
    # conditioning in a FLUX.1 diffusion model with a trained LoRA.
    # Iterates over all LoRA checkpoints in a folder and generates a fixed number of
    # morphing series per checkpoint. Uses an interleaved (round-robin) strategy so that
    # progress is balanced across all checkpoints and the script can be stopped at any
    # time without losing work on any single checkpoint.
    #
    # Two workflow variants are supported, selected via workflow_type:
    #
    #   "double_lora":
    #       Loads the LoRA twice (once per class) and uses a ModelMergeSimple node
    #       in combination with a ConditioningAverage node. This is the workflow that
    #       generated all synthetic images reported in the paper.
    #
    #   "single_lora":
    #       Loads the LoRA once and uses only the ConditioningAverage node. This is
    #       the recommended workflow for new work.
    #
    # The two workflows share the same node IDs for the morph ratio (23), the sampler
    # seed (11), and the SaveImage prefix (14). Only the LoRA loader nodes differ:
    # the double-LoRA workflow uses nodes 5 and 6, the single-LoRA workflow uses node 5.
    def __init__(self, workflow_type: str = WORKFLOW_TYPE) -> None:

        # ===== Workflow selection =====
        if workflow_type not in ("double_lora", "single_lora"):
            raise ValueError(
                f"workflow_type must be 'double_lora' or 'single_lora', got '{workflow_type}'"
            )
        self.workflow_type = workflow_type

        # ===== Generation parameters =====
        self.num_images = NUM_IMAGES
        self.num_series = NUM_SERIES

        # ===== Morph range =====
        self.morph_start = MORPH_START
        self.morph_end = MORPH_END

        # ===== Seed mode =====
        self.seed_mode = SEED_MODE

        # ===== LoRA strength =====
        self.lora_strength_model = LORA_STRENGTH_MODEL
        self.lora_strength_clip = LORA_STRENGTH_CLIP

        # ===== ComfyUI server =====
        self.server_url = COMFYUI_SERVER_URL

        # ===== Workflow template =====
        # The workflow JSON is resolved relative to this script's location.
        workflow_filename = (
            "workflow_double_lora.json"
            if self.workflow_type == "double_lora"
            else "workflow_single_lora.json"
        )
        self.template_path = Path(__file__).parent / workflow_filename

        # ===== Checkpoint configuration =====
        self.checkpoints_base_dir = Path(CHECKPOINTS_DIR)
        self.checkpoint_format = CHECKPOINT_FORMAT
        self.sort_checkpoints_forward = True
        self.checkpoint_files: List[Tuple[int, str, Path]] = []

        # ===== Output configuration =====
        self.base_output_dir = Path(OUTPUT_DIR)

        # ===== Statistics =====
        self.generation_stats = {
            "total_requested": 0,
            "total_generated": 0,
            "failed_generations": 0,
            "checkpoint_stats": {}
        }
        self.completed_series_per_checkpoint: Dict[int, int] = {}

        # ===== Load workflow template =====
        with open(self.template_path, 'r') as f:
            self.workflow_template = json.load(f)

    #############################################################################################################
    # METHODS

    # Scan the checkpoint directory for .safetensors files and extract step numbers.
    def scan_checkpoints(self) -> None:
        if not self.checkpoints_base_dir.exists():
            raise FileNotFoundError(f"Checkpoints directory not found: {self.checkpoints_base_dir}")

        self.checkpoint_files = []

        if self.checkpoint_format.lower() == "comfyui":
            step_pattern = r"-step(\d+)\.safetensors$"
        else:
            step_pattern = r"-(\d{6})\.safetensors$"

        print(f"Scanning for checkpoints in: {self.checkpoints_base_dir}")
        print(f"  Format:  {self.checkpoint_format.upper()}")
        print(f"  Pattern: {step_pattern}")

        numbered_checkpoints = []
        for filename in os.listdir(self.checkpoints_base_dir):
            if not filename.endswith('.safetensors'):
                continue
            match = re.search(step_pattern, filename)
            if match:
                step = int(match.group(1))
                full_path = self.checkpoints_base_dir / filename
                numbered_checkpoints.append((step, filename, full_path))
                print(f"  Found: {filename} (step {step})")

        if not numbered_checkpoints:
            raise FileNotFoundError(
                f"No checkpoint files found in {self.checkpoints_base_dir} "
                f"matching pattern {step_pattern}"
            )

        self.checkpoint_files = sorted(
            numbered_checkpoints,
            key=lambda x: x[0],
            reverse=not self.sort_checkpoints_forward
        )

        for step, _, _ in self.checkpoint_files:
            self.completed_series_per_checkpoint[step] = 0

        print(f"Found {len(self.checkpoint_files)} checkpoints total")
        order = "ascending" if self.sort_checkpoints_forward else "descending"
        print(f"  Order: {order} - step {self.checkpoint_files[0][0]} "
              f"-> {self.checkpoint_files[-1][0]}")

    # Create the output folder for a given checkpoint step.
    def create_checkpoint_folder(self, checkpoint_step: int) -> Optional[Path]:
        checkpoint_folder = f"checkpoint_{checkpoint_step}"
        folder_path = self.base_output_dir / checkpoint_folder
        try:
            folder_path.mkdir(parents=True, exist_ok=True)
            return folder_path
        except Exception as e:
            print(f"  Could not create checkpoint folder {folder_path}: {e}")
            return None

    # Generate a single morphing series for a specific checkpoint.
    def generate_single_series(
        self,
        checkpoint_step: int,
        checkpoint_filename: str,
        series_seed: int,
        checkpoint_output_folder: Path
    ) -> int:
        morph_ratios = []
        if self.num_images > 1:
            step_size = (self.morph_end - self.morph_start) / (self.num_images - 1)
            morph_ratios = [self.morph_start + (i * step_size) for i in range(self.num_images)]
        else:
            morph_ratios = [self.morph_start]

        successful_generations = 0

        for i, ratio in enumerate(morph_ratios):
            if self.seed_mode == "random_per_image":
                image_seed = random.randint(0, 2**32 - 1)
            else:
                image_seed = series_seed

            filename_prefix = (
                f"checkpoint_{checkpoint_step}/"
                f"s{image_seed}_ckpt{checkpoint_step}_{i+1:02d}_r{ratio:.2f}"
            )

            success = self.generate_single_image(
                ratio, image_seed, filename_prefix,
                checkpoint_filename, checkpoint_step, i
            )

            if success:
                successful_generations += 1
                if i == 0:
                    if self.seed_mode == "fixed_per_series":
                        print(f"   Series queued (series seed: {series_seed})")
                    else:
                        print(f"   Series queued (random seeds per image)")
            else:
                print(f"   Failed at ratio {ratio:.2f}")
                break

        return successful_generations

    # Update the LoRA loader nodes in the workflow with the current checkpoint.
    #
    # The node IDs differ between the two workflow types:
    #   double_lora: LoRA loaders are at nodes 5 and 6
    #   single_lora: LoRA loader is at node 5
    def _update_lora_nodes(self, workflow: dict, checkpoint_filename: str) -> None:
        if self.workflow_type == "double_lora":
            # Node 5: first LoRA loader
            workflow["5"]["inputs"]["lora_name"] = checkpoint_filename
            workflow["5"]["inputs"]["strength_model"] = self.lora_strength_model
            workflow["5"]["inputs"]["strength_clip"] = self.lora_strength_clip
            # Node 6: second LoRA loader
            workflow["6"]["inputs"]["lora_name"] = checkpoint_filename
            workflow["6"]["inputs"]["strength_model"] = self.lora_strength_model
            workflow["6"]["inputs"]["strength_clip"] = self.lora_strength_clip
        else:
            # Node 5: single LoRA loader
            workflow["5"]["inputs"]["lora_name"] = checkpoint_filename
            workflow["5"]["inputs"]["strength_model"] = self.lora_strength_model
            workflow["5"]["inputs"]["strength_clip"] = self.lora_strength_clip

    # Generate a single image with a specific morph ratio, seed, and checkpoint.
    def generate_single_image(
        self,
        morph_ratio: float,
        seed: int,
        filename_prefix: str,
        checkpoint_filename: str,
        checkpoint_step: int,
        image_index: int = 0
    ) -> bool:

        workflow = json.loads(json.dumps(self.workflow_template))

        # Node 23: morph ratio (shared between both workflows)
        workflow["23"]["inputs"]["value"] = morph_ratio

        # Node 11: sampler seed (shared between both workflows)
        workflow["11"]["inputs"]["seed"] = seed

        # LoRA loader nodes (workflow-type specific)
        self._update_lora_nodes(workflow, checkpoint_filename)

        # Node 14: SaveImage filename prefix (shared between both workflows)
        workflow["14"]["inputs"]["filename_prefix"] = filename_prefix

        try:
            data = {"prompt": workflow}
            response = requests.post(f"{self.server_url}/prompt", json=data, timeout=30)

            if response.status_code == 200:
                if checkpoint_step not in self.generation_stats["checkpoint_stats"]:
                    self.generation_stats["checkpoint_stats"][checkpoint_step] = {
                        "total": 0,
                        "per_ratio": {}
                    }

                rounded_ratio = round(morph_ratio, 2)
                if rounded_ratio not in self.generation_stats["checkpoint_stats"][checkpoint_step]["per_ratio"]:
                    self.generation_stats["checkpoint_stats"][checkpoint_step]["per_ratio"][rounded_ratio] = 0

                self.generation_stats["checkpoint_stats"][checkpoint_step]["per_ratio"][rounded_ratio] += 1
                self.generation_stats["checkpoint_stats"][checkpoint_step]["total"] += 1
                self.generation_stats["total_generated"] += 1
                return True
            else:
                print(f"   API Error: {response.status_code} - {response.text[:200]}")
                return False

        except requests.exceptions.Timeout:
            print(f"   Timeout error - ComfyUI not responding")
            return False
        except requests.exceptions.ConnectionError:
            print(f"   Connection error - Is ComfyUI running?")
            return False
        except Exception as e:
            print(f"   Error: {e}")
            return False

    # Check if ComfyUI is running and accessible.
    def check_comfyui_connection(self) -> bool:
        try:
            response = requests.get(f"{self.server_url}/queue", timeout=5)
            if response.status_code == 200:
                print("ComfyUI is running and accessible")
                return True
            else:
                print(f"ComfyUI returned status code: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("Cannot connect to ComfyUI.")
            print("   Make sure it is running with: python main.py --listen")
            return False
        except requests.exceptions.Timeout:
            print("ComfyUI connection timeout")
            return False
        except Exception as e:
            print(f"Error checking ComfyUI: {e}")
            return False

    # Generate morphing series interleaved across all checkpoints.
    def generate_all_checkpoints_interleaved(self) -> dict:
        self.scan_checkpoints()

        if not self.checkpoint_files:
            print("No checkpoints found. Cannot generate images.")
            return self.generation_stats

        print(f"\nMulti-Checkpoint Morph Series Generator (INTERLEAVED MODE)")
        print(f"Workflow type: {self.workflow_type.upper()}")
        print("=" * 70)
        print(f"  Checkpoints folder:           {self.checkpoints_base_dir}")
        print(f"  Checkpoints found:            {len(self.checkpoint_files)}")
        print(f"  Target series per checkpoint: {self.num_series}")
        print(f"  Total series to generate:     {len(self.checkpoint_files) * self.num_series}")
        print(f"  Total images:                 {len(self.checkpoint_files) * self.num_series * self.num_images}")
        print(f"  Morph range:                  {self.morph_start} -> {self.morph_end}")
        print(f"  Seed mode:                    {self.seed_mode.upper()}")
        print(f"  LoRA strength (model/clip):   {self.lora_strength_model:.2f}/{self.lora_strength_clip:.2f}")
        print("=" * 70)

        print("\nChecking ComfyUI connection...")
        if not self.check_comfyui_connection():
            print("Generation cancelled.")
            return self.generation_stats

        checkpoint_folders = {}
        valid_checkpoints = []

        for step, filename, full_path in self.checkpoint_files:
            folder = self.create_checkpoint_folder(step)
            if folder:
                checkpoint_folders[step] = folder
                valid_checkpoints.append((step, filename, full_path))
            else:
                print(f"Cannot create folder for checkpoint {step}. Skipping.")

        if not valid_checkpoints:
            print("No valid checkpoints to process.")
            return self.generation_stats

        self.checkpoint_files = valid_checkpoints

        print(f"\nStarting INTERLEAVED generation across {len(self.checkpoint_files)} checkpoints...")
        print(f"   One series per checkpoint in round-robin fashion.")
        print(f"   Press Ctrl+C to stop gracefully at any time.\n")

        total_series_target = len(self.checkpoint_files) * self.num_series
        completed_series_total = 0

        try:
            while completed_series_total < total_series_target:
                for step, filename, full_path in self.checkpoint_files:
                    current_completed = self.completed_series_per_checkpoint[step]
                    if current_completed >= self.num_series:
                        continue

                    series_seed = random.randint(0, 2**32 - 1)

                    print(f"\n[{time.strftime('%H:%M:%S')}] Checkpoint {step} - "
                          f"Series {current_completed + 1}/{self.num_series}")
                    if self.seed_mode == "fixed_per_series":
                        print(f"   Series seed: {series_seed}")

                    successful = self.generate_single_series(
                        step, filename, series_seed, checkpoint_folders[step]
                    )

                    if successful == self.num_images:
                        self.completed_series_per_checkpoint[step] += 1
                        completed_series_total += 1
                        self.generation_stats["total_requested"] += self.num_images

                        percent_complete = (completed_series_total / total_series_target) * 100
                        print(f"   Progress: {completed_series_total}/{total_series_target} "
                              f"series ({percent_complete:.1f}%)")
                    else:
                        print(f"   Series incomplete. Will retry later.")

                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nGeneration interrupted by user!")
            print(f"   Completed {completed_series_total} out of {total_series_target} series.")

        total_requested = sum(self.completed_series_per_checkpoint.values()) * self.num_images
        self.generation_stats["total_requested"] = total_requested
        self.generation_stats["failed_generations"] = (
            total_requested - self.generation_stats["total_generated"]
        )

        return self.generation_stats

    # Print final summary of generation.
    def print_final_summary(self) -> None:
        print(f"\n{'='*70}")
        print("GENERATION COMPLETE")
        print("=" * 70)
        print(f"  Workflow type:          {self.workflow_type.upper()}")
        print(f"  Checkpoints processed:  {len(self.checkpoint_files)}")
        print(f"  Total series requested: {sum(self.completed_series_per_checkpoint.values())}")
        print(f"  Total images queued:    {self.generation_stats['total_generated']}")
        print(f"  Failed:                 {self.generation_stats['failed_generations']}")
        print(f"  Seed mode:              {self.seed_mode.upper()}")

        print(f"\nPer-checkpoint progress:")
        for step, filename, _ in self.checkpoint_files:
            completed = self.completed_series_per_checkpoint.get(step, 0)
            print(f"   Checkpoint {step}: {completed}/{self.num_series} series "
                  f"({completed * self.num_images} images)")

        print(f"\nCheck ComfyUI's output folder for your generated images!")
        print(f"   Base output: {self.base_output_dir}")

        print(f"\nFile naming convention:")
        print("   checkpoint_{step}/s{seed}_ckpt{step}_{image:02d}_r{ratio}.png")
        if self.seed_mode == "random_per_image":
            print("   Note: Each image in a series has a different seed (visible in filename)")


#############################################################################################################
# MAIN

def main() -> None:
    print("Multi-Checkpoint Fibroblast WT/KO Morph Series Generator")
    print("=" * 70)
    print("Generates morphing series for multiple LoRA checkpoints")
    print("INTERLEAVED MODE: One series per checkpoint in round-robin")
    print("   - Balanced progress across all checkpoints")
    print("   - Stop anytime without losing progress on any checkpoint")
    print("=" * 70)

    try:
        generator = FibroblastMorphGenerator(workflow_type=WORKFLOW_TYPE)
        print(f"\nCheckpoint folder: {generator.checkpoints_base_dir}")
        generator.generate_all_checkpoints_interleaved()
        generator.print_final_summary()

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("   Please check the paths in the CONFIGURATION block at the top of this file.")
    except json.JSONDecodeError as e:
        print(f"\nJSON Error: {e}")
        print("   The workflow template file may be corrupted or have invalid JSON.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("\nMake sure:")
        print("   - ComfyUI is running with the API enabled (--listen flag)")
        print("   - The workflow template path is correct")
        print("   - The checkpoints directory contains .safetensors files")
        print("   - The requests module is installed: pip install requests")


if __name__ == "__main__":
    main()