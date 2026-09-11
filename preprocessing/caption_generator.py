# Diffusion-Based Phenotypic Extrapolation
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: MIT

"""
Caption modes:
  1. CELL_LINE_ONLY: "WT_JG cells, grayscale"
  2. PHENOTYPE_ONLY: "wildtype cells, grayscale" 
  3. BOTH: "WT_JG wildtype cells, grayscale"
"""

# ===== Standard Library Imports =====
from pathlib import Path
from enum import Enum
# ===== Own Modules =====
from settings import setting

class CaptionMode(Enum):
    """Enumeration of caption generation modes."""
    CELL_LINE_ONLY = "cell_line_only"
    PHENOTYPE_ONLY = "phenotype_only"
    BOTH = "both"

class CaptionGenerator:

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the caption generator.
    # Args:
    #   image_folder (Path): Path to the folder containing all images (can have subfolders)
    #   output_folder (Path, optional): Path to output folder. Defaults to output/ in settings.
    #   mode (CaptionMode): The caption generation mode
    #   overwrite_existing (bool): If True, overwrite existing caption files
    #   verbose (bool): Print progress messages
    def __init__(
        self,
        image_folder: Path,
        output_folder: Path = None,
        mode: CaptionMode = CaptionMode.PHENOTYPE_ONLY,
        overwrite_existing: bool = True,
        verbose: bool = True
    ) -> None:
        
        self.image_folder = Path(image_folder)
        self.output_folder = Path(output_folder) if output_folder else setting['pth_output']
        self.mode = mode
        self.overwrite_existing = overwrite_existing
        self.verbose = verbose

        # Cell line definitions from settings
        self.wt_lines = setting.get('preproc_caption_wt_lines', ["WT_1618-02", "WT_JG", "WT_JT", "WT_KM", "WT_MS"])
        self.ko_lines = setting.get('preproc_caption_ko_lines', ["KO_1096-01", "KO_1618-01", "KO_BR2986", "KO_BR3075"])
        self.all_lines = self.wt_lines + self.ko_lines

        # Image extensions to process
        self.image_extensions = setting.get('preproc_caption_image_extensions', ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'])

        # Statistics
        self.stats = {
            'total_images': 0,
            'created': 0,
            'overwritten': 0,
            'skipped': 0,
            'token_errors': 0,
            'other_errors': 0,
            'line_stats': {line: 0 for line in self.all_lines},
            'line_stats_unknown': 0
        }

    #############################################################################################################
    # METHODS

    # Determine the output folder structure based on input.
    # Args:
    #   image_path (Path): Path to the image file
    # Returns:
    #   Path: Output directory for the caption file
    def _get_output_subfolder(self, image_path: Path) -> Path:
        # Get relative path from image folder
        rel_path = image_path.relative_to(self.image_folder)

        # If image is directly in image folder (no parent), use "captions"
        if len(rel_path.parent.parts) == 0 or rel_path.parent == Path('.'):
            return self.output_folder / "captions"

        # Otherwise, use the folder name from the input structure
        return self.output_folder / rel_path.parent

    # Extract cell line name from filename by checking if it starts with any known cell line.
    # Args:
    #   filename (str): Image filename
    # Returns:
    #   str or None: Clean cell line name or None if not found
    def _extract_cell_line_from_filename(self, filename: str) -> str:
        sorted_lines = sorted(self.all_lines, key=len, reverse=True)
        for cell_line in sorted_lines:
            if filename.startswith(cell_line):
                return cell_line
        return None

    # Extract cell line and phenotype from filename.
    # Args:
    #   filename (str): Image filename
    # Returns:
    #   tuple: (cell_line, phenotype) or (None, None) if not found
    def _extract_tokens_from_filename(self, filename: str) -> tuple:
        cell_line = self._extract_cell_line_from_filename(filename)

        if not cell_line:
            return None, None

        if cell_line in self.ko_lines:
            phenotype = "knockout"
        elif cell_line in self.wt_lines:
            phenotype = "wildtype"
        else:
            phenotype = None

        return cell_line, phenotype

    # Generate the caption text based on the selected mode.
    # Args:
    #   cell_line (str): Cell line name
    #   phenotype (str): Phenotype name (wildtype/knockout)
    # Returns:
    #   str or None: Caption text
    def _generate_caption(self, cell_line: str, phenotype: str) -> str:
        if not cell_line or not phenotype:
            return None

        if self.mode == CaptionMode.CELL_LINE_ONLY:
            caption_parts = [f"{cell_line} cells"]
        elif self.mode == CaptionMode.PHENOTYPE_ONLY:
            caption_parts = [f"{phenotype} cells"]
        elif self.mode == CaptionMode.BOTH:
            caption_parts = [f"{cell_line} {phenotype} cells"]
        else:
            caption_parts = [f"{cell_line} {phenotype} cells"]

        caption_parts.append("grayscale")
        return ", ".join(caption_parts)

    # Generate caption files for all images in the folder.
    # Returns:
    #   dict: Statistics of the operation
    def generate_caption_files(self) -> dict:
        print(f"{'='*70}")
        print("CAPTION FILE GENERATOR")
        print(f"{'='*70}")
        print(f"Image folder: {self.image_folder}")
        print(f"Output folder: {self.output_folder}")
        print(f"Caption mode: {self.mode.value}")
        print(f"Overwrite existing: {self.overwrite_existing}")
        print(f"{'='*70}")

        if not self.image_folder.exists():
            print(f"Error: Folder '{self.image_folder}' does not exist!")
            return self.stats

        if not self.image_folder.is_dir():
            print(f"Error: '{self.image_folder}' is not a directory!")
            return self.stats

        # Get ALL image files recursively
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(self.image_folder.rglob(f"*{ext}"))
            image_files.extend(self.image_folder.rglob(f"*{ext.upper()}"))

        image_files = [f for f in set(image_files) if f.is_file()]

        if not image_files:
            print(f"No image files found in '{self.image_folder}' or its subfolders")
            return self.stats

        self.stats['total_images'] = len(image_files)
        print(f"Total images found: {len(image_files)} (recursive search)")
        print(f"Known cell lines: {len(self.all_lines)}")
        print("-" * 70)

        for image_file in image_files:
            cell_line, phenotype = self._extract_tokens_from_filename(image_file.name)

            if not cell_line or not phenotype:
                self.stats['token_errors'] += 1
                if self.verbose:
                    print(f"  Warning: Could not extract tokens from '{image_file.name}'")
                self.stats['line_stats_unknown'] += 1
                continue

            if cell_line in self.stats['line_stats']:
                self.stats['line_stats'][cell_line] += 1
            else:
                self.stats['line_stats_unknown'] += 1

            caption_text = self._generate_caption(cell_line, phenotype)

            if not caption_text:
                self.stats['other_errors'] += 1
                if self.verbose:
                    print(f"  Error: Could not generate caption for '{image_file.name}'")
                continue

            # Determine output folder
            output_subfolder = self._get_output_subfolder(image_file)
            output_subfolder.mkdir(parents=True, exist_ok=True)

            # Create caption file path
            caption_file_path = output_subfolder / image_file.with_suffix('.txt').name

            # Check if caption file already exists
            if caption_file_path.exists():
                if not self.overwrite_existing:
                    self.stats['skipped'] += 1
                    continue
                else:
                    self.stats['overwritten'] += 1
            else:
                self.stats['created'] += 1

            try:
                caption_file_path.write_text(caption_text, encoding='utf-8')

                if self.verbose and (self.stats['created'] + self.stats['overwritten']) <= 5:
                    rel_path = image_file.relative_to(self.image_folder)
                    action = "Overwriting" if caption_file_path.exists() else "Creating"
                    print(f"  {action}: '{rel_path}' -> '{caption_file_path}'")

            except Exception as e:
                print(f"  Error writing caption for '{image_file.name}': {e}")
                self.stats['other_errors'] += 1

        self._print_summary()
        return self.stats

    # Print summary statistics.
    def _print_summary(self) -> None:
        print("-" * 70)
        print("Summary:")
        print(f"  Total images found: {self.stats['total_images']}")
        print(f"  Created: {self.stats['created']} new caption files")
        print(f"  Overwritten: {self.stats['overwritten']} existing files")
        print(f"  Skipped: {self.stats['skipped']} files (already existed)")
        print(f"  Token extraction errors: {self.stats['token_errors']} files")
        print(f"  Other errors: {self.stats['other_errors']} files")
        print(f"  Successfully processed: {self.stats['created'] + self.stats['overwritten']} images")
        print(f"  Output folder: {self.output_folder}")

        # Print cell line statistics
        print("\nCell Line Distribution:")
        print("-" * 40)
        for line in sorted(self.all_lines):
            count = self.stats['line_stats'][line]
            if count > 0:
                print(f"  {line}: {count} images")
        if self.stats['line_stats_unknown'] > 0:
            print(f"  Unknown: {self.stats['line_stats_unknown']} images")


    #############################################################################################################
    # CALL

    def __call__(self) -> dict:
        return self.generate_caption_files()

########
# MAIN #
########

# Main entry point for the caption generator.
# Called either directly or from the preprocessing menu.
def main() -> None:
    # Load settings from main settings
    mode_str = setting.get('preproc_caption_mode', 'phenotype_only')
    mode_map = {
        'cell_line_only': CaptionMode.CELL_LINE_ONLY,
        'phenotype_only': CaptionMode.PHENOTYPE_ONLY,
        'both': CaptionMode.BOTH
    }
    mode = mode_map.get(mode_str, CaptionMode.PHENOTYPE_ONLY)
    overwrite = setting.get('preproc_caption_overwrite', True)

    # Use input folder as source
    image_folder = setting['pth_input']
    output_folder = setting['pth_output']

    print(f"\nCaption Generator")
    print("=" * 60)
    print(f"Input folder: {image_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Mode: {mode.value}")
    print(f"Overwrite: {overwrite}")
    print("=" * 60)

    generator = CaptionGenerator(
        image_folder=image_folder,
        output_folder=output_folder,
        mode=mode,
        overwrite_existing=overwrite
    )
    generator()


if __name__ == "__main__":
    main()