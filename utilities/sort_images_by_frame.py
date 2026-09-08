# ===== Standard Library Imports =====
import shutil
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
# ===== Own Modules =====
from settings import setting

class ImageOrganizerByFrame:

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the image organizer by frame.
    # Args:
    #   format (str): Image naming format - "flux" or "stylegan". Defaults to "flux".
    #   verbose (bool): Print progress messages. Defaults to True.
    def __init__(self, format: str = "flux", verbose: bool = True) -> None:

        # Paths from settings
        self.input_folder = setting['pth_input']
        self.output_folder = setting['pth_output']

        self.format = format
        self.verbose = verbose

        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)

    #############################################################################################################
    # METHODS

    # Extract frame number from filename based on selected format.
    # Args:
    #   filename (Path or str): The filename to process
    # Returns:
    #   Optional[str]: Frame number if found, None otherwise
    def extract_frame_number(self, filename) -> Optional[str]:
        if isinstance(filename, Path):
            filename_str = filename.stem
        else:
            filename_str = str(Path(filename).stem)

        if self.format == "flux":
            # New Flux pattern: s{seed}_ckpt{checkpoint}_{frame}_r{morph_ratio}_{number}
            match = re.match(r"s\d+_ckpt\d+_(\d+)_r[\d\.\-]+_\d+_", filename_str)
            if match:
                return match.group(1)

            # Old Flux pattern: s{seed}_{frame}_fib_morph
            match = re.match(r"s\d+_(\d+)_fib_morph", filename_str)
            if match:
                return match.group(1)

        elif self.format == "stylegan":
            # StyleGAN pattern: {series}_{frame}
            match = re.match(r"\d{4}_(\d{2})$", filename_str)
            if match:
                return match.group(1)

        return None

    # Get statistics about available images before organizing.
    # Returns:
    #   Tuple[Dict[str, int], int]: (frame_stats, skipped_count)
    def get_statistics(self) -> Tuple[Dict[str, int], int]:
        image_files = list(self.input_folder.rglob("*.png"))
        frame_stats = {}
        skipped_count = 0

        for img_path in image_files:
            frame = self.extract_frame_number(img_path)
            if frame is not None:
                frame_stats[frame] = frame_stats.get(frame, 0) + 1
            else:
                skipped_count += 1

        return frame_stats, skipped_count

    # Print detailed statistics about found images.
    def print_statistics(self) -> None:
        stats, skipped = self.get_statistics()

        print(f"\n{'='*60}")
        print(f"IMAGE STATISTICS ({self.format.upper()} FORMAT)")
        print(f"{'='*60}")

        total_found = sum(stats.values())
        print(f"\nTotal matching images: {total_found}")
        print(f"Skipped (non-matching): {skipped}")

        if stats:
            print(f"\nFrame distribution:")
            print("-" * 40)
            sorted_stats = sorted(stats.items(), key=lambda x: int(x[0]))
            for frame, count in sorted_stats:
                print(f"  Frame {frame}: {count:5d} images")

            if self.format == "flux":
                print(f"\nExpected frames: 01-10 (10 frames per series)")
            elif self.format == "stylegan":
                print(f"\nExpected frames: 00-10 (11 frames per series)")

        print(f"{'='*60}")

    # Organize all images into folders by frame number.
    # Returns:
    #   Dict[str, int]: Frame -> image count
    def organize_images_by_frame(self) -> Dict[str, int]:
        image_files = list(self.input_folder.rglob("*.png"))

        if not image_files:
            print("No PNG files found in input folder!")
            return {}

        frame_folders = {}
        processed_count = 0
        skipped_count = 0

        print(f"Organizing images by frame number...")
        print(f"Format: {self.format.upper()}")

        for img_path in image_files:
            frame = self.extract_frame_number(img_path)

            if frame is not None:
                frame_folder = self.output_folder / f"frame_{frame}"
                frame_folder.mkdir(exist_ok=True)

                shutil.copy2(img_path, frame_folder / img_path.name)

                frame_folders[frame] = frame_folders.get(frame, 0) + 1
                processed_count += 1
            else:
                skipped_count += 1

        print(f"\nOrganization complete!")
        print(f"Processed {processed_count} images")
        print(f"Skipped {skipped_count} images (did not match naming pattern)")
        print(f"Created {len(frame_folders)} frame folders:")

        sorted_frames = sorted(frame_folders.items(), key=lambda x: int(x[0]))
        for frame, count in sorted_frames:
            print(f"  frame_{frame}: {count} images")

        return frame_folders

    #############################################################################################################
    # CALL

    # Run the image organizer by frame.
    # Returns:
    #   Dict[str, int]: Frame -> image count
    def __call__(self) -> Dict[str, int]:
        return self.organize_images_by_frame()