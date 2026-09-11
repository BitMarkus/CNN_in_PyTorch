# Diffusion-Based Phenotypic Extrapolation
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: MIT

# ===== Own Modules =====
import functions as fn
from settings import setting
from . import czi_export_main, CaptionGenerator, CaptionMode

class PreprocessingMenu:
    # Preprocessing Menu - Collection of data preparation tools.

    #############################################################################################################
    # CONSTRUCTOR

    def __init__(self) -> None:
        pass

    #############################################################################################################
    # METHODS

    # Display the preprocessing submenu and route to the selected option.
    def menu(self) -> None:

        while True:
            print("\n:PREPROCESSING MENU:")
            print("1) Export CZI Mosaic Files (Convert .czi to PNG)")
            print("2) Generate Captions (for LoRA training)")
            print("3) Back to Main Menu")

            choice = fn.input_int("Please choose: ")

            if choice == 1:
                self._run_czi_export()
            elif choice == 2:
                self._run_caption_generator()
            elif choice == 3:
                print("\nReturning to main menu...")
                break
            else:
                print("Not a valid option!")

    # Run the CZI Export.
    def _run_czi_export(self) -> None:
        print("\n:CZI MOSAIC EXPORT:")
        print("  Input: input/ (place .czi files here)")
        print("  Output: output/[czi_filename]/ (PNG images)")
        print("  Note: This converts Zeiss .czi mosaic files to individual training images.")
        print()

        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Operation cancelled.")
            return

        try:
            czi_export_main()
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()

    # Run the Caption Generator.
    def _run_caption_generator(self) -> None:
        print("\n:CAPTION GENERATOR:")
        print("  Input: input/ (images in subfolders or directly)")
        print("  Output: output/[folder_name]/ or output/captions/")
        print("  Note: Generates text captions for LoRA training.")
        print(f"  Mode: {setting.get('preproc_caption_mode', 'phenotype_only')}")
        print()

        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Operation cancelled.")
            return

        try:
            # Load settings
            mode_str = setting.get('preproc_caption_mode', 'phenotype_only')
            mode_map = {
                'cell_line_only': CaptionMode.CELL_LINE_ONLY,
                'phenotype_only': CaptionMode.PHENOTYPE_ONLY,
                'both': CaptionMode.BOTH
            }
            mode = mode_map.get(mode_str, CaptionMode.PHENOTYPE_ONLY)
            overwrite = setting.get('preproc_caption_overwrite', True)

            generator = CaptionGenerator(
                image_folder=setting['pth_input'],
                output_folder=setting['pth_output'],
                mode=mode,
                overwrite_existing=overwrite
            )
            generator()
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()