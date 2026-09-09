# ===== Own Modules =====
import functions as fn
from analysis import ClassAnalyzer, GradCAMAnalyzer, ClassSorter, FIDCalculator, DimRed

class AnalysisMenu:

    #############################################################################################################
    # CONSTRUCTOR

    def __init__(self) -> None:
        pass

    #############################################################################################################
    # METHODS

    # Display the Analysis submenu and route to the selected option.
    # Args:
    #   device (torch.device): Device to run operations on
    def menu(self, device) -> None:
        while True:
            print("\n:ANALYSIS MENU:")
            print("1) Predict Class from Input Folder")
            print("2) GradCAM Analyzer")
            print("3) Class Sorter (High-Confidence Image Selection)")
            print("4) FID Score Calculator")
            print("5) Dimensionality Reduction (UMAP/t-SNE/TriMAP/PaCMAP)")
            print("6) Back to Main Menu")

            choice = fn.input_int("Please choose: ")

            if choice == 1:
                self._predict_from_input_folder(device)
            elif choice == 2:
                self._run_gradcam_analyzer(device)
            elif choice == 3:
                self._run_class_sorter(device)
            elif choice == 4:
                self._run_fid_calculator(device)
            elif choice == 5:
                self._run_dim_red(device)
            elif choice == 6:
                print("\nReturning to main menu...")
                break
            else:
                print("Not a valid option!")

    # Predict Class from Input Folder.
    def _predict_from_input_folder(self, device) -> None:
        print("\n:PREDICT CLASS FROM INPUT FOLDER:")
        print("  Place images to classify in the input/ folder")
        print("  Results will be saved to output/")

        analyzer = ClassAnalyzer(device)
        analyzer.analyze_prediction_folder()

    # Run GradCAM Analyzer.
    def _run_gradcam_analyzer(self, device) -> None:
        print("\n:GradCAM ANALYZER:")
        print("  Input: input/ (place images to analyze)")
        print("  Output: output/gradcam/")
        gradcam = GradCAMAnalyzer(device)
        gradcam()

    # Run Class Sorter.
    def _run_class_sorter(self, device) -> None:
        print("\n:CLASS SORTER:")
        print("  Input: input/ (place images to sort by confidence)")
        print("  Output: output/class_sorter/ (organized by class and confidence)")
        print("  Note: Selects high-confidence images for LoRA training.")
        print()

        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Operation cancelled.")
            return

        try:
            sorter = ClassSorter(device)
            output_dir = sorter.run()
            print(f"\n✅ Class Sorter complete! Output saved to: {output_dir}")
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()

    # Run FID Calculator.
    def _run_fid_calculator(self, device) -> None:
        print("\n:FID SCORE CALCULATOR:")
        print("  Input: input/ (place folders with images to compare)")
        print("  Output: output/fid_results.txt")
        print("  Note: Computes Fréchet Inception Distance between folders.")
        print()

        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Operation cancelled.")
            return

        try:
            # Optional: allow user to specify reference folder
            ref_folder = input("Enter reference folder name (or press Enter for auto-detect): ").strip()
            if not ref_folder:
                ref_folder = None

            fid = FIDCalculator(device, reference_folder=ref_folder)
            fid()
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()

    # Run Dimensionality Reduction.
    def _run_dim_red(self, device) -> None:
        print("\n:DIMENSIONALITY REDUCTION:")
        print("  Input: input/ (images) or data/train/ / data/test/")
        print("  Output: output/dim_red/ (embedding CSV/JSON and plots)")
        print("  Note: Supports UMAP, t-SNE, TriMAP, and PaCMAP.")
        print()

        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Operation cancelled.")
            return

        try:
            dimred = DimRed(device)
            dimred()
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()