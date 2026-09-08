# ===== Standard Library Imports =====
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
# ===== Own Modules =====
from settings import setting

class DatasetSubtractor:

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the dataset subtractor.
    # Args:
    #   dataset_a_name (str): Name of the first dataset folder. Defaults to "dataset_a".
    #   dataset_b_name (str): Name of the second dataset folder. Defaults to "dataset_b".
    #   result_name (str): Name of the result folder. Defaults to "result_dataset".
    #   verbose (bool): Print progress messages. Defaults to True.
    def __init__(
        self,
        dataset_a_name: str = "dataset_a",
        dataset_b_name: str = "dataset_b",
        result_name: str = "result_dataset",
        verbose: bool = True
    ) -> None:
        
        # Paths from settings
        self.input_folder = setting['pth_input']
        self.output_folder = setting['pth_output']

        self.dataset_a_folder = self.input_folder / dataset_a_name
        self.dataset_b_folder = self.input_folder / dataset_b_name
        self.result_folder = self.output_folder / result_name
        self.verbose = verbose

        # Create result folder if it doesn't exist
        self.result_folder.mkdir(parents=True, exist_ok=True)

    #############################################################################################################
    # METHODS

    # Extract the base identifier from filename (everything before '_conf' if present).
    def extract_base_identifier(self, filename) -> str:
        if isinstance(filename, Path):
            filename_str = filename.stem
        else:
            filename_str = str(Path(filename).stem)

        return filename_str.split('_conf')[0]

    # Create a mapping of base identifiers to their file paths.
    def get_image_mapping(self, folder_path: Path) -> Dict[str, List[Path]]:
        mapping = defaultdict(list)

        if not folder_path.exists():
            if self.verbose:
                print(f"Warning: Folder {folder_path} does not exist!")
            return mapping

        image_files = list(folder_path.rglob("*.png"))

        if self.verbose:
            print(f"  Found {len(image_files)} PNG files")

        for img_path in image_files:
            base_identifier = self.extract_base_identifier(img_path.name)
            mapping[base_identifier].append(img_path)

        return mapping

    # Prompt the user to choose the subtraction direction.
    def choose_subtraction_direction(
        self,
        dataset_a_map: Dict,
        dataset_b_map: Dict
    ) -> Tuple[Dict, str, Dict, str]:
        print("\n3. Choose subtraction direction:")
        print("   Which operation do you want to perform?")
        print(f"   1) dataset_a - dataset_b  (subtract dataset_b from dataset_a)")
        print(f"   2) dataset_b - dataset_a  (subtract dataset_a from dataset_b)")

        while True:
            try:
                choice = input("\n   Enter 1 or 2: ").strip()
                if choice == "1":
                    minuend_map = dataset_a_map
                    minuend_name = "dataset_a"
                    subtrahend_map = dataset_b_map
                    subtrahend_name = "dataset_b"
                    print(f"\n   Selected: {minuend_name} - {subtrahend_name}")
                    return minuend_map, minuend_name, subtrahend_map, subtrahend_name
                elif choice == "2":
                    minuend_map = dataset_b_map
                    minuend_name = "dataset_b"
                    subtrahend_map = dataset_a_map
                    subtrahend_name = "dataset_a"
                    print(f"\n   Selected: {minuend_name} - {subtrahend_name}")
                    return minuend_map, minuend_name, subtrahend_map, subtrahend_name
                else:
                    print("   Invalid choice. Please enter 1 or 2.")
            except KeyboardInterrupt:
                print("\n\nOperation cancelled.")
                exit(0)

    # Subtract datasets based on user choice.
    def subtract_datasets(self) -> Dict:
        print("="*80)
        print("DATASET SUBTRACTION TOOL")
        print("="*80)

        print("\n1. Scanning dataset_a folder...")
        dataset_a_map = self.get_image_mapping(self.dataset_a_folder)
        print(f"   Found {len(dataset_a_map)} unique base identifiers")
        total_a_files = sum(len(files) for files in dataset_a_map.values())
        print(f"   Total files: {total_a_files}")

        print("\n2. Scanning dataset_b folder...")
        dataset_b_map = self.get_image_mapping(self.dataset_b_folder)
        print(f"   Found {len(dataset_b_map)} unique base identifiers")
        total_b_files = sum(len(files) for files in dataset_b_map.values())
        print(f"   Total files: {total_b_files}")

        minuend_map, minuend_name, subtrahend_map, subtrahend_name = self.choose_subtraction_direction(
            dataset_a_map, dataset_b_map
        )

        common_identifiers = set(minuend_map.keys()) & set(subtrahend_map.keys())

        print(f"\n4. Set Operations:")
        print(f"   Unique identifiers in {minuend_name}: {len(minuend_map)}")
        print(f"   Unique identifiers in {subtrahend_name}: {len(subtrahend_map)}")
        print(f"   Common identifiers (present in both): {len(common_identifiers)}")

        if len(common_identifiers) == 0:
            print(f"\n⚠️  WARNING: No common identifiers found between datasets!")
            print(f"   There are no images to subtract.")
            print(f"   Result will be a copy of {minuend_name}.")

        result_identifiers = set(minuend_map.keys()) - common_identifiers

        print(f"   Result identifiers ({minuend_name} - common): {len(result_identifiers)}")

        minuend_multiples = {id: files for id, files in minuend_map.items() if len(files) > 1}
        if minuend_multiples:
            print(f"\n   Note: Found {len(minuend_multiples)} identifiers with multiple files in {minuend_name}")

        print(f"\n5. Copying result images...")
        print(f"   Destination: {self.result_folder}")

        stats = {
            'total_copied': 0,
            'identifiers_copied': 0,
            'common_identifiers': len(common_identifiers),
            'files_per_identifier': defaultdict(int),
            'errors': 0,
            'minuend': minuend_name,
            'subtrahend': subtrahend_name,
            'operation': f"{minuend_name} - {subtrahend_name}",
            'result_empty': False
        }

        for identifier in result_identifiers:
            files_to_copy = minuend_map[identifier]
            stats['identifiers_copied'] += 1
            stats['files_per_identifier'][len(files_to_copy)] += 1

            for file_path in files_to_copy:
                try:
                    shutil.copy2(file_path, self.result_folder / file_path.name)
                    stats['total_copied'] += 1
                except Exception as e:
                    if self.verbose:
                        print(f"   Error copying {file_path.name}: {e}")
                    stats['errors'] += 1

        if stats['total_copied'] == 0:
            print(f"\n⚠️  WARNING: Resulting dataset is empty!")
            stats['result_empty'] = True

        return stats

    # Print formatted statistics to console.
    def print_statistics(self, stats: Dict) -> None:
        print("\n" + "="*80)
        print("DATASET SUBTRACTION STATISTICS")
        print("="*80)

        print(f"\nSummary:")
        print(f"  Operation: {stats['operation']}")
        print(f"  Minuend (source): {stats['minuend']}")
        print(f"  Subtrahend (to subtract): {stats['subtrahend']}")
        print(f"  Common identifiers subtracted: {stats['common_identifiers']}")
        print(f"  Result identifiers: {stats['identifiers_copied']}")
        print(f"  Total files copied: {stats['total_copied']}")

        if stats['result_empty']:
            print(f"  ⚠️  RESULT IS EMPTY!")

        if stats['common_identifiers'] == 0:
            print(f"  ⚠️  NO IMAGES SUBTRACTED (no common identifiers)")

        if stats['errors'] > 0:
            print(f"  Errors during copying: {stats['errors']}")

        if stats['files_per_identifier']:
            print(f"\nFiles per identifier distribution:")
            for count in sorted(stats['files_per_identifier'].keys()):
                identifiers = stats['files_per_identifier'][count]
                print(f"  {count} file(s) per identifier: {identifiers} identifier(s)")

        print(f"\nResult dataset saved to: {self.result_folder}")
        print("="*80)

    # Verify that the result dataset doesn't contain images that should have been subtracted.
    def verify_dataset_integrity(self, stats: Dict) -> Tuple[bool, int]:
        print("\n" + "="*80)
        print("DATASET INTEGRITY VERIFICATION")
        print("="*80)

        result_map = self.get_image_mapping(self.result_folder)
        dataset_a_map = self.get_image_mapping(self.dataset_a_folder)
        dataset_b_map = self.get_image_mapping(self.dataset_b_folder)

        if stats['minuend'] == 'dataset_a':
            minuend_map = dataset_a_map
            subtrahend_map = dataset_b_map
        else:
            minuend_map = dataset_b_map
            subtrahend_map = dataset_a_map

        common_identifiers = set(minuend_map.keys()) & set(subtrahend_map.keys())
        overlaps = set(result_map.keys()) & common_identifiers

        if overlaps:
            print(f"❌ FAIL: Found {len(overlaps)} common identifiers in result!")
            print(f"\nOverlapping identifiers (first 10):")
            for identifier in sorted(overlaps)[:10]:
                result_files = [f.name for f in result_map[identifier]]
                minuend_files = [f.name for f in minuend_map[identifier]]
                print(f"  - {identifier}:")
                print(f"    Result: {result_files}")
                print(f"    {stats['minuend']}: {minuend_files}")
            if len(overlaps) > 10:
                print(f"    ... and {len(overlaps) - 10} more")
            return False, len(overlaps)
        else:
            print("✅ SUCCESS: No common identifiers found in result!")
            print(f"Dataset subtraction ({stats['operation']}) correctly performed.")
            return True, 0

    #############################################################################################################
    # CALL

    def __call__(self) -> Dict:
        statistics = self.subtract_datasets()
        self.print_statistics(statistics)
        integrity_ok, overlap_count = self.verify_dataset_integrity(statistics)

        if integrity_ok:
            if statistics['result_empty']:
                print(f"\n⚠️  COMPLETED WITH WARNINGS: Result dataset is empty!")
            elif statistics['common_identifiers'] == 0:
                print(f"\n⚠️  COMPLETED WITH WARNINGS: No images were subtracted!")
                print(f"   Result is a copy of {statistics['minuend']}.")
            else:
                print(f"\n✅ SUCCESS: Dataset subtraction complete!")
                print(f"   Operation: {statistics['operation']}")
                print(f"   Files: {statistics['total_copied']}")
                print(f"   Location: {self.result_folder}")
        else:
            print(f"\n❌ ERROR: Dataset has {overlap_count} overlapping identifiers!")
            print("   The result contains images that should have been subtracted.")

        return statistics