# ===== Standard Library Imports =====
from pathlib import Path
import json
# ===== Third-Party Imports =====
import torch
# ===== Own Modules =====
from dataset import Dataset
from dataset_gen import DatasetGenerator
from model import CNN_Model
from train import Train
from settings import setting
import functions as fn


class AutoCrossValidation:

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the automatic cross-validation handler.
    # Performs leave-one-WT-line-out AND leave-one-KO-line-out cross-validation.
    # For N WT lines and M KO lines, runs N × M independent training runs.
    # Args:
    #   device (torch.device): Device to run training on
    def __init__(self, device: torch.device) -> None:

        # Passed parameters
        self.device = device

        # Settings parameters
        self.wt_lines = setting['wt_lines']
        self.ko_lines = setting['ko_lines']
        self.acv_results_dir = setting['pth_acv_results']
        self.data_dir = setting['pth_data']
        self.class_list = setting["classes"]

        # Validation split settings
        self.val_from_train_split = setting["ds_val_from_train_split"]
        self.val_from_test_split = setting["ds_val_from_test_split"]

        # Objects
        self.ds_gen = DatasetGenerator(mode="acv")
        self.ds = Dataset()
        self.cnn_wrapper = None

    #############################################################################################################
    # METHODS

    # Record which images were used for validation vs testing.
    # Only records REAL images (filters out synthetic images).
    # Args:
    #   dataset_idx (int): Index of the current dataset/fold
    def _record_val_test_split(self, dataset_idx: int) -> None:
        if self.val_from_test_split is False:
            return

        try:
            val_loader = self.ds.ds_test_for_val
            test_loader = self.ds.ds_test_for_test

            if not all(hasattr(loader, 'sampler') and hasattr(loader.sampler, 'indices')
                       for loader in [val_loader, test_loader]):
                raise ValueError("DataLoaders don't have proper sampler indices")

            dataset = val_loader.dataset

            if hasattr(dataset, 'dataset'):
                full_dataset = dataset.dataset
                val_indices = [dataset.indices[i] for i in val_loader.sampler.indices]
                test_indices = [dataset.indices[i] for i in test_loader.sampler.indices]
            else:
                full_dataset = dataset
                val_indices = list(val_loader.sampler.indices)
                test_indices = list(test_loader.sampler.indices)

            val_indices_set = set(val_indices)
            test_indices_set = set(test_indices)

            true_val_indices = val_indices_set - test_indices_set
            true_test_indices = test_indices_set - val_indices_set

            if len(val_indices_set & test_indices_set) > 0:
                print(f"Warning: Corrected {len(val_indices_set & test_indices_set)} overlapping images")

            val_images = [full_dataset.samples[i][0] for i in true_val_indices]
            test_images = [full_dataset.samples[i][0] for i in true_test_indices]

            def is_synthetic_image(path) -> bool:
                filename = Path(path).name
                return filename.startswith('s') and filename[1:2].isdigit()

            original_val_count = len(val_images)
            original_test_count = len(test_images)

            real_val_images = [p for p in val_images if not is_synthetic_image(p)]
            real_test_images = [p for p in test_images if not is_synthetic_image(p)]

            synthetic_val_filtered = original_val_count - len(real_val_images)
            synthetic_test_filtered = original_test_count - len(real_test_images)

            if synthetic_val_filtered > 0 or synthetic_test_filtered > 0:
                print(f"Filtered out {synthetic_val_filtered} synthetic images from validation set")
                print(f"Filtered out {synthetic_test_filtered} synthetic images from test set")

            if len(real_val_images) == 0:
                print("WARNING: No real images left in validation set!")
            if len(real_test_images) == 0:
                print("WARNING: No real images left in test set!")

            split_info = {
                'validation': {
                    'WT': sorted(list({Path(p).name for p in real_val_images if 'WT' in str(p)})),
                    'KO': sorted(list({Path(p).name for p in real_val_images if 'KO' in str(p)}))
                },
                'test': {
                    'WT': sorted(list({Path(p).name for p in real_test_images if 'WT' in str(p)})),
                    'KO': sorted(list({Path(p).name for p in real_test_images if 'KO' in str(p)}))
                },
                'metadata': {
                    'total_images_found': original_val_count + original_test_count,
                    'real_images_used': len(real_val_images) + len(real_test_images),
                    'synthetic_filtered': synthetic_val_filtered + synthetic_test_filtered,
                    'note': 'Only real images are recorded for validation/testing'
                }
            }

            output_path = Path(self.acv_results_dir) / f"dataset_{dataset_idx}" / "split_info.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(split_info, f, indent=2)

            print(f"Recorded {len(real_val_images)} real validation images and {len(real_test_images)} real test images")

        except Exception as e:
            print(f"Failed to record splits: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    # Reinitialize the model by creating a fresh instance.
    # Ensures each cross-validation fold starts with a clean model.
    def reset_model(self) -> None:
        print("Creating fresh model instance for new training...")
        self.cnn_wrapper.model = self.cnn_wrapper.load_model(self.device)
        self.cnn_wrapper.model = self.cnn_wrapper.model.to(self.device)

    #############################################################################################################
    # CALL

    # Run the complete cross-validation.
    # Iterates over all WT/KO line combinations, trains a model on each combination,
    # and saves results to the acv_results directory.
    def __call__(self) -> None:
        print("\nCleaning up old train and test data...")
        self.ds_gen.cleanup(self.data_dir)
        print("Cleanup finished.")

        self.acv_results_dir.mkdir(parents=True, exist_ok=True)

        configs = self.ds_gen.get_dataset_configs()

        for config in configs:
            self.ds = Dataset()
            self.cnn_wrapper = CNN_Model()
            self.reset_model()

            self.ds_gen.cleanup(self.data_dir)
            torch.cuda.empty_cache()

            ##################
            # Create dataset #
            ##################

            print(f"\n>> PROCESSING DATASET {config['dataset_idx']} OF {len(configs)}:")
            print(f"Training data source: {self.ds_gen.training_data_source}")

            print(f"Cell line for testing WT group: {config['test_wt']}")
            print(f"Cell line for testing KO group: {config['test_ko']}")

            print(f"\n> Create dataset {config['dataset_idx']}...")
            dataset_dir = self.ds_gen.generate_dataset(**config)

            checkpoint_dir = dataset_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            plot_dir = dataset_dir / "plots"
            plot_dir.mkdir(exist_ok=True)
            print(f"Dataset {config['dataset_idx']} successfully created.")

            ######################
            # Load train dataset #
            ######################

            print(f"\n> Load dataset {config['dataset_idx']} for training...")

            self.ds.validate_validation_settings()

            self.ds.load_training_dataset()

            if self.val_from_test_split is not False:
                self.ds.load_test_dataset()
                self._record_val_test_split(config['dataset_idx'])

            self.ds.print_dataset_info()
            if self.ds.ds_loaded:
                print(f"Dataset {config['dataset_idx']} successfully loaded.")
                print(f"Number training images/batches: {self.ds.num_train_img}/{self.ds.num_train_batches}")
                print(f"Number validation images/batches: {self.ds.num_val_img}/{self.ds.num_val_batches}")

            ####################
            # Train on dataset #
            ####################

            self.train = Train(self.cnn_wrapper, self.ds, self.device, dataset_idx=config['dataset_idx'])
            print(f"\n> Start training on dataset {config['dataset_idx']}...")
            self.train.train(checkpoint_dir, plot_dir)
            print(f"\nTraining on dataset {config['dataset_idx']} successfully finished.")

            ################
            # Testing Loop #
            ################

            if self.val_from_test_split is not False and self.val_from_test_split != 1.0:
                split_file = Path(self.acv_results_dir) / f"dataset_{config['dataset_idx']}" / "split_info.json"

                if split_file.exists():
                    with open(split_file, 'r') as f:
                        split_info = json.load(f)

                    real_test_images = set(split_info['test']['WT'] + split_info['test']['KO'])

                    if len(real_test_images) > 0:
                        print(f"\n>>> Evaluating ALL checkpoints on TEST set ({len(real_test_images)} images)...")

                        success = self.ds.load_real_test_dataset_only(real_test_images)
                        if success:
                            test_dataset = self.ds.ds_test_real_only

                            checkpoint_dir = Path(self.acv_results_dir) / f"dataset_{config['dataset_idx']}" / "checkpoints"
                            checkpoint_files = sorted(checkpoint_dir.glob("*.pt"))

                            print(f"Found {len(checkpoint_files)} checkpoints to evaluate on test set")

                            from settings import setting
                            pretrained_str = "pretr" if setting["cnn_is_pretrained"] else "scratch"
                            model_name = setting["cnn_type"]

                            for checkpoint_path in checkpoint_files:
                                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                                self.cnn_wrapper.model.load_state_dict(checkpoint['model_state_dict'])
                                self.cnn_wrapper.model.to(self.device)
                                self.cnn_wrapper.model.eval()

                                epoch = checkpoint.get('epoch', 'unknown')
                                if epoch != 'unknown':
                                    epoch = f"{epoch+1:02d}"

                                acc = checkpoint.get('accuracy', 'unknown')
                                if acc != 'unknown':
                                    acc = int(acc * 100)

                                print(f"\n  > Testing checkpoint epoch {epoch} (val_acc={acc}%)...")

                                _, cm = self.cnn_wrapper.predict(test_dataset)

                                chckpt_name = f"ckpt_{pretrained_str}_{model_name}_test_e{epoch}_vacc{acc}_ds{config['dataset_idx']}"

                                fn.plot_confusion_matrix(
                                    {"y": cm.get('y', []), "y_hat": cm.get('y_hat', [])} if isinstance(cm, dict) else cm,
                                    self.class_list,
                                    checkpoint_dir.parent / "plots",
                                    chckpt_name=chckpt_name,
                                    show_plot=False,
                                    save_plot=True
                                )
                                fn.save_confusion_matrix_results(
                                    cm,
                                    self.class_list,
                                    checkpoint_dir.parent / "plots",
                                    chckpt_name=chckpt_name
                                )

                                loaded_results = fn.load_confusion_matrix_results(
                                    checkpoint_dir.parent / "plots",
                                    file_name=chckpt_name
                                )
                                if loaded_results:
                                    print(f"    Test accuracy: {(loaded_results['overall_accuracy']*100):.2f}%")
                                    print(f"    Test WT: {(loaded_results['class_accuracy']['WT']*100):.2f}%")
                                    print(f"    Test KO: {(loaded_results['class_accuracy']['KO']*100):.2f}%")

                                del checkpoint
                                torch.cuda.empty_cache()

                            print(f"\n✅ Finished testing all checkpoints for dataset {config['dataset_idx']}")
                        else:
                            print(f"Warning: Could not load test dataset for dataset {config['dataset_idx']}")
                    else:
                        print(f"No test images found in split_info for dataset {config['dataset_idx']}")
                else:
                    print(f"split_info.json not found for dataset {config['dataset_idx']}")
            else:
                if self.val_from_test_split == 1.0:
                    print(f"\n⚠️  ds_val_from_test_split = 1.0 - No separate test set. Skipping test evaluation.")
                else:
                    print(f"\n⚠️  No test set configured (val_from_test_split = {self.val_from_test_split}). Skipping test evaluation.")

            print(f"Completed dataset {config['dataset_idx']}. Moving to next fold...")

        print("\nCleaning up all temporary data...")
        self.ds_gen.cleanup(self.data_dir)
        print("Cross-validation complete.")