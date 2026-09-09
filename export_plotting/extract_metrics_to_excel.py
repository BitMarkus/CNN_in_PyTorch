# ===== Standard Library Imports =====
from pathlib import Path
from collections import defaultdict
import re
# ===== Third-Party Imports =====
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

class TensorBoardExporter:

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the TensorBoard exporter.
    # Args:
    #   logdir (Path): Path to TensorBoard logs directory
    #   output_file (Path): Path for output Excel file
    #   prob_dir (Path, optional): Path to directory containing probability .npz files
    #   roc_epoch (str or int): Which epoch to use for ROC curves
    #   pr_epoch (str or int): Which epoch to use for PR curves (same options)
    #   mode (str): Detection mode ('auto', 'crossval', 'single')
    def __init__(
        self,
        logdir,
        output_file=None,
        prob_dir=None,
        roc_epoch='composite_score',
        pr_epoch='composite_score',
        mode='auto'
    ) -> None:
        
        self.logdir = Path(logdir)
        self.output_file = Path(output_file) if output_file else Path("./output/train_metrics.xlsx")
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.prob_dir = Path(prob_dir) if prob_dir else None
        self.roc_epoch = roc_epoch
        self.pr_epoch = pr_epoch
        self.mode = mode

        self.runs_data = {}
        self.run_type = None
        self.saved_checkpoints = {}

    #############################################################################################################
    # METHODS

    # Check if run name follows cross-validation pattern (ds01, ds02, etc.).
    def _is_cross_validation_run(self, run_name: str) -> bool:
        return bool(re.match(r'^ds\d{2}$', run_name))

    # Check if run name follows timestamp pattern (YYYYMMDD-HHMMSS).
    def _is_timestamp_run(self, run_name: str) -> bool:
        return bool(re.match(r'^\d{8}-\d{6}$', run_name))

    # Auto-detect whether we're dealing with cross-validation or single training.
    def _detect_run_type(self, runs: list) -> str:
        if self.mode == 'crossval':
            print("✓ Mode forced: CROSS-VALIDATION")
            return 'crossval'
        elif self.mode == 'single':
            print("✓ Mode forced: SINGLE TRAINING")
            return 'single'

        crossval_count = 0
        timestamp_count = 0

        for run_path, run_name, _ in runs:
            if self._is_cross_validation_run(run_name):
                crossval_count += 1
            elif self._is_timestamp_run(run_name):
                timestamp_count += 1

        if crossval_count > 0 and timestamp_count == 0:
            print(f"✓ Auto-detected: CROSS-VALIDATION mode ({crossval_count} dsXX folders)")
            return 'crossval'
        elif timestamp_count > 0 and crossval_count == 0:
            print(f"✓ Auto-detected: SINGLE TRAINING mode ({timestamp_count} timestamp folders)")
            return 'single'
        elif crossval_count > 0 and timestamp_count > 0:
            print(f"⚠ Mixed run types detected! Using CROSS-VALIDATION mode as default.")
            return 'crossval'
        else:
            if list(self.logdir.glob("events.out.tfevents.*")):
                print(f"✓ Auto-detected: SINGLE TRAINING mode (direct event files)")
                return 'single'
            print(f"⚠ Could not determine run type. Using CROSS-VALIDATION mode as default.")
            return 'crossval'

    # Get display name for Excel sheet based on run type.
    def _get_display_name(self, run_name: str) -> str:
        if self.run_type == 'crossval':
            return run_name
        else:
            if self._is_timestamp_run(run_name):
                return run_name[:31]
            return "training_run"

    # Get the probability subdirectory based on run type.
    def _get_probability_subdir(self, run_name: str):
        if not self.prob_dir:
            return None

        if self.run_type == 'crossval':
            # Convert dsXX to dataset_XX
            dataset_num = run_name.replace("ds", "")
            dataset_name = f"dataset_{dataset_num}"

            # NEW structure: dataset_X/logs/probabilities/
            prob_subdir = self.prob_dir / dataset_name / "logs" / "probabilities"
            if prob_subdir.exists():
                return prob_subdir

            # OLD structure: logs/dsXX/probabilities/
            prob_subdir = self.logdir / run_name / "probabilities"
            if prob_subdir.exists():
                return prob_subdir

            # Alternative: self.logdir is the logs folder with dataset_X
            prob_subdir = self.logdir / dataset_name / "logs" / "probabilities"
            if prob_subdir.exists():
                return prob_subdir

            return None
        else:
            # Single training: same as before
            if self.prob_dir.name == "probabilities":
                return self.prob_dir
            prob_subdir = self.logdir / "probabilities"
            if prob_subdir.exists():
                return prob_subdir
            if run_name and self.logdir.name != run_name:
                prob_subdir = self.logdir / run_name / "probabilities"
            else:
                prob_subdir = self.logdir / "probabilities"
            return prob_subdir

    # Get saved checkpoints for a run by looking for .pt files.
    def _get_saved_checkpoints(self, run_path: Path, run_name: str) -> set:
        saved_epochs = set()
        checkpoints_dir = None

        if self.run_type == 'crossval':
            # Try multiple possible locations
            
            # NEW structure: checkpoints are in dataset_X/checkpoints/
            dataset_num = run_name.replace("ds", "")
            dataset_name = f"dataset_{dataset_num}"
            
            # If run_path is the dataset folder
            if run_path.name.startswith("dataset_"):
                checkpoints_dir = run_path / "checkpoints"
            # If run_path is the logs folder
            elif run_path.name == "logs":
                checkpoints_dir = run_path.parent / "checkpoints"
            # Try by dataset name
            if not checkpoints_dir or not checkpoints_dir.exists():
                checkpoints_dir = self.logdir / dataset_name / "checkpoints"
            # Try parent (old structure)
            if not checkpoints_dir or not checkpoints_dir.exists():
                checkpoints_dir = self.logdir.parent / dataset_name / "checkpoints"
            # OLD structure: checkpoints were in the dataset folder
            if not checkpoints_dir or not checkpoints_dir.exists():
                checkpoints_dir = self.logdir / run_name / "checkpoints"
            # Final fallback
            if not checkpoints_dir or not checkpoints_dir.exists():
                checkpoints_dir = self.logdir.parent / run_name / "checkpoints"
        else:
            # Single training: same as before
            checkpoints_dir = run_path.parent.parent / "checkpoints"
            if not checkpoints_dir.exists():
                checkpoints_dir = run_path.parent / "checkpoints"
            if not checkpoints_dir.exists():
                checkpoints_dir = run_path / "checkpoints"

        if not checkpoints_dir or not checkpoints_dir.exists():
            return saved_epochs

        checkpoint_files = list(checkpoints_dir.glob("*.pt"))

        for f in checkpoint_files:
            try:
                match = re.search(r'_e(\d+)', f.stem)
                if match:
                    epoch = int(match.group(1))
                    saved_epochs.add(epoch - 1)
                else:
                    match = re.search(r'(\d+)\.pt$', f.name)
                    if match:
                        epoch = int(match.group(1))
                        saved_epochs.add(epoch - 1)
            except Exception:
                continue

        return saved_epochs

    # Extract metric from events with optional target value.
    @staticmethod
    def _extract_metric_from_events(events, metric_name: str, target_value: str = None):
        steps, values = [], []
        for e in events:
            if isinstance(e.value, dict):
                if target_value and target_value in e.value:
                    steps.append(e.step)
                    values.append(e.value[target_value])
                elif metric_name in e.value:
                    steps.append(e.step)
                    values.append(e.value[metric_name])
            else:
                steps.append(e.step)
                values.append(e.value)
        if not steps:
            return None, None
        d = defaultdict(list)
        for s, v in zip(steps, values):
            d[s].append(v)
        steps_sorted = sorted(d.keys())
        return steps_sorted, [np.mean(d[s]) for s in steps_sorted]

    # Get best epoch by a specific metric.
    def _get_best_epoch_by_metric(self, run_df, metric_name: str):
        if metric_name not in run_df.columns:
            return None

        scores = run_df[metric_name].values
        epochs = run_df['epoch'].values
        valid = ~np.isnan(scores)

        if not np.any(valid):
            return None

        best_idx = np.argmax(scores[valid])
        return int(epochs[valid][best_idx])

    # Find runs in the log directory (supports both old and new structures)
    def find_runs(self) -> list:
        all_event_files = list(self.logdir.rglob("events.out.tfevents.*"))

        if not all_event_files:
            print(f"❌ No event files found in {self.logdir}")
            return []

        folders = defaultdict(list)
        
        for ef in all_event_files:
            parent = ef.parent
            
            # NEW structure: dataset_X/logs/
            if parent.name == "logs" and parent.parent.name.startswith("dataset_"):
                run_name = parent.parent.name.replace("dataset_", "ds")
                folders[parent.parent].append(ef)
            
            # NEW structure (alternative): dataset_X/logs/ with run_name already
            elif parent.name == "logs" and parent.parent.name.startswith("ds"):
                run_name = parent.parent.name
                folders[parent.parent].append(ef)
            
            # OLD structure: logs/dsXX/
            elif parent.name.startswith("ds") and parent.parent.name == "logs":
                run_name = parent.name
                folders[parent].append(ef)
            
            # Single training: direct event files
            elif parent.name == "logs" and parent.parent.name not in ["cross_validation", "acv_results"]:
                # Single training - use the timestamp folder name
                run_name = parent.parent.name
                folders[parent.parent].append(ef)
            
            # Fallback: use the folder name
            else:
                run_name = parent.name
                folders[parent].append(ef)

        runs = []
        for folder_path, files in folders.items():
            # Determine run name from folder structure
            if folder_path.name.startswith("dataset_"):
                run_name = folder_path.name.replace("dataset_", "ds")
            elif folder_path.name.startswith("ds") and folder_path.parent.name == "logs":
                run_name = folder_path.name
            else:
                run_name = folder_path.name
            runs.append((folder_path, run_name, files))

        if runs:
            self.run_type = self._detect_run_type(runs)

        return runs

    # Extract all scalar data from a run's event files.
    def extract_run_data(self, run_path: Path, run_name: str):
        run_path = Path(run_path)
        event_files = list(run_path.rglob("events.out.tfevents.*"))
        if not event_files:
            return None, None

        saved_checkpoints = self._get_saved_checkpoints(run_path, run_name)
        self.saved_checkpoints[run_name] = saved_checkpoints

        all_metrics = defaultdict(list)

        for ef in event_files:
            try:
                ea = event_accumulator.EventAccumulator(
                    str(ef),
                    size_guidance=event_accumulator.STORE_EVERYTHING_SIZE_GUIDANCE
                )
                ea.Reload()

                for tag in ea.Tags()['scalars']:
                    events = ea.Scalars(tag)

                    if tag == 'Metrics/F1':
                        s, v = self._extract_metric_from_events(events, 'Macro', 'Macro')
                        if s:
                            all_metrics['F1_macro'].extend(zip(s, v))
                        s, v = self._extract_metric_from_events(events, 'Weighted', 'Weighted')
                        if s:
                            all_metrics['F1_weighted'].extend(zip(s, v))
                        continue

                    s = [e.step for e in events]
                    v = [e.value for e in events]
                    if not s:
                        continue

                    tag_lower = tag.lower()
                    if 'loss' in tag_lower:
                        mname = 'Loss_train' if 'train' in tag_lower else ('Loss_val' if 'val' in tag_lower else None)
                    elif 'accuracy' in tag_lower or 'acc' in tag_lower:
                        if 'train' in tag_lower:
                            mname = 'Accuracy_train_standard' if 'standard' in tag_lower else 'Accuracy_train_weighted'
                        elif 'val' in tag_lower:
                            mname = 'Accuracy_val_standard' if 'standard' in tag_lower else 'Accuracy_val_weighted'
                        elif 'class' in tag_lower:
                            if 'std' in tag_lower:
                                mname = 'Class_Accuracy_StdDev'
                            elif 'min' in tag_lower:
                                mname = 'Min_Class_Accuracy'
                            else:
                                mname = f'Accuracy_per_class_{tag.split("/")[-1]}'
                        else:
                            continue
                    elif 'balanced' in tag_lower and 'accuracy' in tag_lower:
                        mname = 'Balanced_accuracy'
                    elif 'f1' in tag_lower:
                        mname = 'F1_weighted' if 'weighted' in tag_lower else 'F1_macro'
                    elif 'auc' in tag_lower:
                        mname = 'AUC_overall' if 'class' not in tag_lower else f'AUC_{tag.split("/")[-1]}'
                    elif 'ap' in tag_lower or 'average_precision' in tag_lower:
                        mname = 'AP_overall' if 'class' not in tag_lower else f'AP_{tag.split("/")[-1]}'
                    elif 'lr' in tag_lower or 'learning_rate' in tag_lower:
                        mname = 'Learning_rate'
                    elif 'composite' in tag_lower:
                        mname = 'Composite_score'
                    elif 'gpu_memory' in tag_lower:
                        mname = 'GPU_memory_usage'
                    elif 'class_count' in tag_lower and 'data' in tag_lower:
                        mname = f'Class_count_{tag.split("/")[-1]}'
                    elif 'class_weight' in tag_lower and 'data' in tag_lower:
                        mname = f'Class_weight_{tag.split("/")[-1]}'
                    else:
                        continue

                    if mname:
                        all_metrics[mname].extend(zip(s, v))

            except Exception:
                continue

        if not all_metrics:
            return None, saved_checkpoints

        all_epochs_0indexed = sorted(set().union(*[set(dict(pairs).keys()) for pairs in all_metrics.values()]))
        all_epochs_1indexed = [ep + 1 for ep in all_epochs_0indexed]

        temp_data = {'epoch': all_epochs_1indexed}
        for mname, pairs in all_metrics.items():
            d = dict(pairs)
            temp_data[mname] = [d.get(ep_0indexed, np.nan) for ep_0indexed in all_epochs_0indexed]

        df = pd.DataFrame(temp_data)

        # Calculate balanced accuracy if not present (fallback)
        if 'Balanced_accuracy' not in df.columns:
            per_class_cols = [col for col in df.columns if col.startswith('Accuracy_per_class_')]
            if per_class_cols:
                bal_acc_values = []
                for idx, epoch in enumerate(all_epochs_1indexed):
                    class_accs = []
                    for col in per_class_cols:
                        val = df[col].values[idx]
                        if not np.isnan(val):
                            class_accs.append(val)
                    if class_accs:
                        bal_acc_values.append(np.mean(class_accs))
                    else:
                        bal_acc_values.append(np.nan)
                df['Balanced_accuracy'] = bal_acc_values

        # Build checkpoint column
        checkpoint_column = []
        for ep_1indexed, ep_0indexed in zip(all_epochs_1indexed, all_epochs_0indexed):
            cell_value = ''
            if ep_0indexed in saved_checkpoints:
                cell_value = 'X'
            checkpoint_column.append(cell_value)

        df.insert(1, 'Checkpoint', checkpoint_column)

        cols = ['epoch', 'Checkpoint'] + [c for c in df.columns if c not in ['epoch', 'Checkpoint']]
        df = df[cols]

        return df, saved_checkpoints

    # Load probability data for a run.
    def _load_probabilities_for_run(self, run_df, epoch_choice, run_name=None):
        if not self.prob_dir or not self.prob_dir.exists():
            return None, None, None, None

        prob_subdir = self._get_probability_subdir(run_name)
        if not prob_subdir or not prob_subdir.exists():
            prob_subdir = self.prob_dir

        npz_files = sorted(prob_subdir.glob("probabilities_epoch_*.npz"))

        if not npz_files and run_name:
            npz_files = sorted(self.prob_dir.glob(f"*{run_name}*_probabilities_epoch_*.npz"))

        if not npz_files:
            alt_dir = self.logdir / run_name / "probabilities" if run_name else None
            if alt_dir and alt_dir.exists():
                npz_files = sorted(alt_dir.glob("probabilities_epoch_*.npz"))

        if not npz_files:
            return None, None, None, None

        epoch_files = []
        for f in npz_files:
            try:
                ep = int(f.stem.split('_')[-1])
                epoch_files.append((ep, f))
            except:
                continue

        if not epoch_files:
            return None, None, None, None

        epoch_files.sort(key=lambda x: x[0])

        selected_epoch = None
        selected_file = None

        if epoch_choice == 'last':
            selected_epoch, selected_file = epoch_files[-1]
        elif epoch_choice == 'composite_score':
            best_epoch = self._get_best_epoch_by_metric(run_df, 'Composite_score')
            if best_epoch is not None:
                for ep, f in epoch_files:
                    if ep == best_epoch:
                        selected_epoch, selected_file = ep, f
                        break
            if selected_epoch is None:
                selected_epoch, selected_file = epoch_files[-1]
        elif epoch_choice == 'balanced_accuracy':
            best_epoch = self._get_best_epoch_by_metric(run_df, 'Balanced_accuracy')
            if best_epoch is not None:
                for ep, f in epoch_files:
                    if ep == best_epoch:
                        selected_epoch, selected_file = ep, f
                        break
            if selected_epoch is None:
                selected_epoch, selected_file = epoch_files[-1]
        else:
            try:
                target = int(epoch_choice)
                for ep, f in epoch_files:
                    if ep == target:
                        selected_epoch, selected_file = ep, f
                        break
                if selected_epoch is None:
                    selected_epoch, selected_file = epoch_files[-1]
            except:
                selected_epoch, selected_file = epoch_files[-1]

        data = np.load(selected_file)
        probs = data['probabilities']
        labels = data['labels']
        class_names = data['classes'].tolist() if 'classes' in data else [f'Class_{i}' for i in range(probs.shape[1])]

        return probs, labels, class_names, selected_epoch

    # Compute ROC curves.
    @staticmethod
    def _compute_roc(probs, labels, class_names):
        curves = []
        for i, cls in enumerate(class_names):
            y_true = (labels == i).astype(int)
            y_score = probs[:, i]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            curves.append((cls, roc_auc, fpr, tpr))
        return curves

    # Compute PR curves.
    @staticmethod
    def _compute_pr(probs, labels, class_names):
        curves = []
        for i, cls in enumerate(class_names):
            y_true = (labels == i).astype(int)
            y_score = probs[:, i]
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            curves.append((cls, ap, recall, precision))
        return curves

    # Export all data to Excel with charts.
    def export_with_charts(self) -> None:
        print("=" * 60)
        print("TENSORBOARD EXPORTER – 3-COLUMN CHART GRID")
        print("=" * 60)
        print(f"Logdir: {self.logdir}")
        print(f"Output: {self.output_file}")
        if self.prob_dir:
            print(f"Probability dir: {self.prob_dir}")
        print(f"ROC epoch selection: {self.roc_epoch}")
        print(f"PR epoch selection: {self.pr_epoch}")

        runs = self.find_runs()
        if not runs:
            print("No runs found.")
            return

        print("\nExtracting scalar metrics from each run...")

        for run_path, run_name, event_files in runs:
            display_name = self._get_display_name(run_name)
            print(f"  {run_name} -> sheet '{display_name}'")
            df, saved_checkpoints = self.extract_run_data(run_path, run_name)
            if df is not None and len(df) > 0:
                self.runs_data[display_name] = (df, run_name, saved_checkpoints)
                print(f"    -> {len(df)} epochs, {len(df.columns)-1} metrics")
                if saved_checkpoints:
                    print(f"    -> Saved checkpoints at epochs: {sorted(saved_checkpoints)}")
            else:
                print(f"    -> No data extracted")

        if not self.runs_data:
            print("No scalar data to export.")
            return

        print("\nWriting Excel file...")
        with pd.ExcelWriter(self.output_file, engine='xlsxwriter') as writer:
            workbook = writer.book

            for display_name, (df, original_run_name, saved_checkpoints) in self.runs_data.items():
                sheet_name = display_name[:31].replace('[', '_').replace(']', '_').replace(':', '_')
                sheet_name = sheet_name.replace('*', '_').replace('?', '_').replace('/', '_')
                print(f"\n--- Run: {original_run_name} -> sheet '{sheet_name}'")

                # Write scalar table
                df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0, startcol=0)
                worksheet = writer.sheets[sheet_name]

                note = "Checkpoint column: 'X' = checkpoint saved"
                worksheet.write(0, len(df.columns), note, workbook.add_format({'italic': True, 'color': '#666666', 'font_size': 8}))

                for i, col in enumerate(df.columns):
                    max_len = max(df[col].astype(str).str.len().max(), len(col)) + 2
                    worksheet.set_column(i, i, min(max_len, 30))

                # Raw ROC/PR data tables
                scalar_cols = len(df.columns)
                roc_start_col = scalar_cols + 1
                pr_start_col = None
                roc_curves = None
                pr_curves = None
                detail_row_roc = None
                detail_row_pr = None
                roc_selected_epoch = None
                pr_selected_epoch = None

                if self.prob_dir and self.prob_dir.exists():
                    # Load probabilities for ROC
                    probs, labels, class_names, roc_selected_epoch = self._load_probabilities_for_run(
                        df, self.roc_epoch, original_run_name
                    )
                    if probs is not None:
                        roc_curves = self._compute_roc(probs, labels, class_names)
                        header_text = f"ROC CURVES (RAW DATA) - Epoch {roc_selected_epoch + 1}" if roc_selected_epoch is not None else "ROC CURVES (RAW DATA)"
                        worksheet.write(0, roc_start_col, header_text, workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'}))
                        worksheet.write(1, roc_start_col, "Class", workbook.add_format({'bold': True}))
                        worksheet.write(1, roc_start_col + 1, "AUC", workbook.add_format({'bold': True}))
                        for i, (cls, auc_val, _, _) in enumerate(roc_curves):
                            worksheet.write(2 + i, roc_start_col, cls)
                            worksheet.write(2 + i, roc_start_col + 1, auc_val)
                        detail_row_roc = 2 + len(roc_curves) + 2
                        col_offset = 0
                        for cls, auc_val, fpr, tpr in roc_curves:
                            worksheet.write(detail_row_roc, roc_start_col + col_offset, f"{cls} (AUC={auc_val:.3f})",
                                            workbook.add_format({'bold': True}))
                            worksheet.write(detail_row_roc + 1, roc_start_col + col_offset, "FPR")
                            worksheet.write(detail_row_roc + 1, roc_start_col + col_offset + 1, "TPR")
                            for j, (x, y) in enumerate(zip(fpr, tpr)):
                                worksheet.write(detail_row_roc + 2 + j, roc_start_col + col_offset, x)
                                worksheet.write(detail_row_roc + 2 + j, roc_start_col + col_offset + 1, y)
                            col_offset += 3
                        roc_columns_used = col_offset
                        pr_start_col = roc_start_col + roc_columns_used + 1

                    # Load probabilities for PR
                    probs_pr, labels_pr, class_names_pr, pr_selected_epoch = self._load_probabilities_for_run(
                        df, self.pr_epoch, original_run_name
                    )
                    if probs_pr is not None:
                        pr_curves = self._compute_pr(probs_pr, labels_pr, class_names_pr)
                        header_text = f"PRECISION-RECALL CURVES (RAW DATA) - Epoch {pr_selected_epoch + 1}" if pr_selected_epoch is not None else "PRECISION-RECALL CURVES (RAW DATA)"
                        worksheet.write(0, pr_start_col, header_text, workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'}))
                        worksheet.write(1, pr_start_col, "Class", workbook.add_format({'bold': True}))
                        worksheet.write(1, pr_start_col + 1, "AP", workbook.add_format({'bold': True}))
                        for i, (cls, ap_val, _, _) in enumerate(pr_curves):
                            worksheet.write(2 + i, pr_start_col, cls)
                            worksheet.write(2 + i, pr_start_col + 1, ap_val)
                        detail_row_pr = 2 + len(pr_curves) + 2
                        col_offset = 0
                        for cls, ap_val, recall, precision in pr_curves:
                            worksheet.write(detail_row_pr, pr_start_col + col_offset, f"{cls} (AP={ap_val:.3f})",
                                            workbook.add_format({'bold': True}))
                            worksheet.write(detail_row_pr + 1, pr_start_col + col_offset, "Recall")
                            worksheet.write(detail_row_pr + 1, pr_start_col + col_offset + 1, "Precision")
                            for j, (x, y) in enumerate(zip(recall, precision)):
                                worksheet.write(detail_row_pr + 2 + j, pr_start_col + col_offset, x)
                                worksheet.write(detail_row_pr + 2 + j, pr_start_col + col_offset + 1, y)
                            col_offset += 3

                # Chart grid
                metric_cols = [c for c in df.columns if c not in ['epoch', 'Checkpoint']]
                chart_start_row = len(df) + 2
                num_columns = 3
                col_spacing = 5
                row_spacing = 16
                chart_width = 500
                chart_height = 300

                all_charts = []
                for metric in metric_cols:
                    all_charts.append(('metric', metric))
                if roc_curves is not None:
                    all_charts.append(('roc', None))
                if pr_curves is not None:
                    all_charts.append(('pr', None))

                for idx, (chart_type, metric_name) in enumerate(all_charts):
                    col_idx = idx % num_columns
                    row_idx = idx // num_columns
                    row = chart_start_row + row_idx * row_spacing
                    col = col_idx * col_spacing

                    if chart_type == 'metric':
                        metric_pos = df.columns.get_loc(metric_name)
                        num_epochs = len(df)

                        chart = workbook.add_chart({'type': 'line'})
                        chart.add_series({
                            'name': metric_name,
                            'categories': [sheet_name, 1, 0, num_epochs, 0],
                            'values': [sheet_name, 1, metric_pos, num_epochs, metric_pos],
                            'line': {'color': 'black', 'width': 1.5},
                            'marker': {'type': 'circle', 'size': 4,
                                       'border': {'color': 'black'},
                                       'fill': {'color': 'black'}}
                        })
                        chart.set_title({'name': metric_name})
                        chart.set_x_axis({'name': 'Epoch', 'position_axis': 'on_tick', 'min': 1, 'max': num_epochs})
                        chart.set_legend({'none': True})
                        chart.set_size({'width': chart_width, 'height': chart_height})
                        worksheet.insert_chart(row, col, chart)

                    elif chart_type == 'roc' and roc_curves is not None and detail_row_roc is not None:
                        roc_chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
                        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                        col_offset = 0
                        for idx_c, (cls, auc_val, fpr, tpr) in enumerate(roc_curves):
                            data_len = len(fpr)
                            x_range = [sheet_name, detail_row_roc + 2, roc_start_col + col_offset,
                                       detail_row_roc + 2 + data_len - 1, roc_start_col + col_offset]
                            y_range = [sheet_name, detail_row_roc + 2, roc_start_col + col_offset + 1,
                                       detail_row_roc + 2 + data_len - 1, roc_start_col + col_offset + 1]
                            roc_chart.add_series({
                                'name': f"{cls} (AUC={auc_val:.3f})",
                                'categories': x_range,
                                'values': y_range,
                                'line': {'color': colors[idx_c % len(colors)], 'width': 2},
                                'marker': {'type': 'none'}
                            })
                            col_offset += 3

                        chart_title = f'ROC Curves (Epoch {roc_selected_epoch + 1})' if roc_selected_epoch is not None else 'ROC Curves'
                        roc_chart.set_title({'name': chart_title})
                        roc_chart.set_x_axis({'name': 'False Positive Rate', 'min': 0, 'max': 1})
                        roc_chart.set_y_axis({'name': 'True Positive Rate', 'min': 0, 'max': 1})
                        roc_chart.set_legend({'position': 'bottom'})
                        roc_chart.set_size({'width': chart_width, 'height': chart_height})
                        worksheet.insert_chart(row, col, roc_chart)

                    elif chart_type == 'pr' and pr_curves is not None and detail_row_pr is not None:
                        pr_chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
                        col_offset = 0
                        for idx_c, (cls, ap_val, recall, precision) in enumerate(pr_curves):
                            data_len = len(recall)
                            x_range = [sheet_name, detail_row_pr + 2, pr_start_col + col_offset,
                                       detail_row_pr + 2 + data_len - 1, pr_start_col + col_offset]
                            y_range = [sheet_name, detail_row_pr + 2, pr_start_col + col_offset + 1,
                                       detail_row_pr + 2 + data_len - 1, pr_start_col + col_offset + 1]
                            pr_chart.add_series({
                                'name': f"{cls} (AP={ap_val:.3f})",
                                'categories': x_range,
                                'values': y_range,
                                'line': {'color': colors[idx_c % len(colors)], 'width': 2},
                                'marker': {'type': 'none'}
                            })
                            col_offset += 3

                        chart_title = f'Precision-Recall Curves (Epoch {pr_selected_epoch + 1})' if pr_selected_epoch is not None else 'Precision-Recall Curves'
                        pr_chart.set_title({'name': chart_title})
                        pr_chart.set_x_axis({'name': 'Recall', 'min': 0, 'max': 1})
                        pr_chart.set_y_axis({'name': 'Precision', 'min': 0, 'max': 1})
                        pr_chart.set_legend({'position': 'bottom'})
                        pr_chart.set_size({'width': chart_width, 'height': chart_height})
                        worksheet.insert_chart(row, col, pr_chart)

                worksheet.freeze_panes(1, 0)

        print(f"\n✅ Export complete! File saved to: {self.output_file.absolute()}")


    #############################################################################################################
    # CALL

    def __call__(self) -> None:
        self.export_with_charts()