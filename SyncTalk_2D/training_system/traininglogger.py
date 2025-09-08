# traininglogger.py

import os
from datetime import datetime
import json
import time

class TrainingLogger:
    def __init__(self, log_dir: str = "./train_logs"):
        """Initialize training logger that writes to file."""
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Original timestamped files
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
        self.metrics_file = os.path.join(log_dir, f"training_metrics_{timestamp}.json")
        
        # Latest files (always overwritten)
        self.latest_log_file = os.path.join(log_dir, "00_latest_logs.txt")
        self.latest_metrics_file = os.path.join(log_dir, "00_latest_metrics.json")
        
        self.metrics_history = []
        
        # Write header to both files
        header_text = (
            "="*80 + "\n" +
            f"Training Log Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" +
            "="*80 + "\n\n"
        )
        
        # Write to timestamped log
        with open(self.log_file, 'w') as f:
            f.write(header_text)
        
        # Write to latest log (overwrite)
        with open(self.latest_log_file, 'w') as f:
            f.write(header_text)
        
        # Clear latest metrics file
        with open(self.latest_metrics_file, 'w') as f:
            json.dump([], f)
    
    def _write_to_both_logs(self, text: str):
        """Helper method to write text to both log files."""
        # Write to timestamped log
        with open(self.log_file, 'a') as f:
            f.write(text)
        
        # Write to latest log
        with open(self.latest_log_file, 'a') as f:
            f.write(text)
    
    def reset_losses(self):
        """
        Resets and returns new loss accumulator dictionaries for a new epoch.
        """
        losses_G = {
            'gan': 0.0, 
            'l1': 0.0, 
            'perceptual': 0.0, 
            'sync': 0.0, 
            'skin_texture': 0.0, 
            'temporal': 0.0,  # Temporal loss included
            'total': 0.0, 
            'count': 0
        }
        losses_D = {
            'real': 0.0, 
            'fake': 0.0, 
            'r1': 0.0, 
            'total': 0.0, 
            'count': 0
        }
        return losses_G, losses_D
    
    def log_epoch(self, epoch: int, losses_G: dict, losses_D: dict, 
                  scheduler_G, epoch_start_time: float, stage_name: str):
        """
        Processes and logs statistics for a completed epoch.

        This is a convenience wrapper around log_epoch_summary. It calculates
        elapsed time and learning rate, determines the stage, and then calls
        the detailed summary logger.

        Args:
            epoch (int): The completed epoch number (e.g., epoch + 1).
            losses_G (dict): Accumulated generator losses for the epoch.
            losses_D (dict): Accumulated discriminator losses for the epoch.
            scheduler_G: The learning rate scheduler for the generator.
            epoch_start_time (float): The timestamp (from time.time()) when the epoch started.
            stage_name (str): The name of the current training stage (e.g., "Stage 1").
        """
        # 1. Check if any training was done to prevent division by zero
        if losses_G['count'] == 0:
            print(f"Warning: log_epoch called for epoch {epoch} but no batches were processed. Skipping log.")
            return

        # 2. Calculate final values for the epoch
        elapsed_time = time.time() - epoch_start_time
        current_lr = scheduler_G.get_last_lr()[0]
        
        # Determine if it's a warmup/initial stage
        # You can customize this check based on your stage names
        is_warmup = '1' in stage_name or 'Warmup' in stage_name.capitalize()

        # 3. Call the detailed summary function with the prepared data
        self.log_epoch_summary(
            epoch=epoch,
            losses_G=losses_G,
            losses_D=losses_D,
            lr=current_lr,
            is_warmup=is_warmup,
            elapsed_time=elapsed_time
        )

    def log_iteration(self, epoch: int, iteration: int, losses_G: dict, losses_D: dict):
        """Log detailed iteration information."""
        log_text = f"\n[Epoch {epoch}, Iteration {iteration}] Loss Breakdown:\n"
        log_text += "-"*60 + "\n"
        
        # Generator losses
        log_text += "Generator Losses:\n"
        if losses_G['count'] > 0:
            log_text += f"  GAN:          {losses_G.get('gan', 0)/losses_G['count']:8.3f}\n"
            log_text += f"  L1:           {losses_G.get('l1', 0)/losses_G['count']:8.3f}\n"
            log_text += f"  Perceptual:   {losses_G.get('perceptual', 0)/losses_G['count']:8.3f}\n"
            log_text += f"  Sync:         {losses_G.get('sync', 0)/losses_G['count']:8.3f}\n"
            if 'skin_texture' in losses_G:  # Handle skin texture loss
                log_text += f"  Skin Texture: {losses_G.get('skin_texture', 0)/losses_G['count']:8.3f}\n"
            if 'temporal' in losses_G:  # Handle temporal loss
                log_text += f"  Temporal:     {losses_G.get('temporal', 0)/losses_G['count']:8.3f}\n"
            log_text += f"  Total:        {losses_G.get('total', 0)/losses_G['count']:8.3f}\n"
        
        # Discriminator losses
        if losses_D['count'] > 0:
            log_text += "\nDiscriminator Losses:\n"
            log_text += f"  Real:         {losses_D.get('real', 0)/losses_D['count']:8.3f}\n"
            log_text += f"  Fake:         {losses_D.get('fake', 0)/losses_D['count']:8.3f}\n"
            if 'r1' in losses_D:
                log_text += f"  R1:           {losses_D.get('r1', 0)/losses_D['count']:8.3f}\n"
            log_text += f"  Total:        {losses_D.get('total', 0)/losses_D['count']:8.3f}\n"
        log_text += "-"*60 + "\n"
        
        self._write_to_both_logs(log_text)
        
    def log_epoch_summary(self, epoch: int, losses_G: dict, losses_D: dict, 
                          lr: float, is_warmup: bool, elapsed_time: float):
        """Log comprehensive epoch summary information."""
        # --- 1. Calculate Averages ---
        avg_g_total = losses_G['total'] / losses_G['count'] if losses_G['count'] > 0 else 0
        avg_d_total = losses_D['total'] / losses_D['count'] if losses_D['count'] > 0 else 0
        
        # --- 2. Build the Core Metrics Dictionary ---
        metrics = {
            'epoch': epoch,
            'g_loss_total': avg_g_total,
            'd_loss_total': avg_d_total,
            'lr': lr,
            'is_warmup': is_warmup,
            'elapsed_time_seconds': elapsed_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # --- 3. Add Detailed and Calculated Metrics ---
        
        # Detailed Generator Losses (Averaged)
        if losses_G['count'] > 0:
            # Include all losses including temporal
            g_losses_avg = {
                'gan': losses_G['gan'] / losses_G['count'],
                'l1': losses_G['l1'] / losses_G['count'],
                'perceptual': losses_G['perceptual'] / losses_G['count'],
                'sync': losses_G['sync'] / losses_G['count'],
                'skin_texture': losses_G['skin_texture'] / losses_G['count'],
                'temporal': losses_G.get('temporal', 0) / losses_G['count']  # Include temporal
            }
            metrics['g_losses_avg'] = g_losses_avg
            
            # Actual Percentage Contributions
            actual_percentages = {}
            if avg_g_total > 0:
                for name, value in g_losses_avg.items():
                    actual_percentages[name] = (value / avg_g_total) * 100
            
            if actual_percentages:
                metrics['g_loss_percentages'] = actual_percentages

        # Detailed Discriminator Losses (Averaged)
        if losses_D['count'] > 0:
            metrics['d_losses_avg'] = {
                'real': losses_D['real'] / losses_D['count'],
                'fake': losses_D['fake'] / losses_D['count'],
                'r1': losses_D['r1'] / losses_D['count']
            }
        
        self.metrics_history.append(metrics)
        
        # --- 4. Create Human-Readable Summary Text ---
        summary_text = f"\n{'='*80}\n"
        summary_text += f"EPOCH {epoch} SUMMARY {'[WARMUP]' if is_warmup else ''}\n"
        summary_text += f"{'='*80}\n"
        summary_text += f"Time Elapsed: {elapsed_time:.2f} seconds\n"
        summary_text += f"Learning Rate: {lr:.6f}\n"
        summary_text += f"Average Generator Loss: {avg_g_total:.4f}\n"
        
        # Add detailed breakdown of generator losses
        if 'g_losses_avg' in metrics:
            summary_text += "  Generator Loss Components:\n"
            for loss_name, loss_value in metrics['g_losses_avg'].items():
                summary_text += f"    {loss_name.capitalize():12s}: {loss_value:.4f}\n"
        
        # Add the percentage contributions to the text log
        if 'g_loss_percentages' in metrics:
            summary_text += "  Percentage Contributions:\n"
            for loss_name, percentage in metrics['g_loss_percentages'].items():
                # Highlight if temporal loss is active
                if loss_name == 'temporal' and percentage > 0:
                    summary_text += f"    {loss_name.capitalize():12s}: {percentage:5.1f}% (Active)\n"
                else:
                    summary_text += f"    {loss_name.capitalize():12s}: {percentage:5.1f}%\n"

        if not is_warmup and avg_d_total > 0:
            summary_text += f"Average Discriminator Loss: {avg_d_total:.4f}\n"
            if 'd_losses_avg' in metrics:
                summary_text += "  Discriminator Loss Components:\n"
                for loss_name, loss_value in metrics['d_losses_avg'].items():
                    summary_text += f"    {loss_name.capitalize():12s}: {loss_value:.4f}\n"
                    
        summary_text += f"{'='*80}\n\n"
        
        # Write the text summary to its log file
        self._write_to_both_logs(summary_text)
        
        # --- 5. Write Machine-Readable JSON Output ---
        
        # Write the full history to the main metrics file
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Safely write only the LATEST epoch's metrics to the 'latest' file
        temp_latest_file = self.latest_metrics_file + '.tmp'
        with open(temp_latest_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        # Atomically replace the old file with the new one
        os.replace(temp_latest_file, self.latest_metrics_file)
    
    def log_message(self, message: str):
        """Log a general message."""
        log_text = f"\n[{datetime.now().strftime('%H:%M:%S')}] {message}\n"
        self._write_to_both_logs(log_text)