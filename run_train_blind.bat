@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d "%~dp0"
call conda activate unpaired_sr
python train_blind.py --auto_resume -opt options/blind_pdm/train_hat_l_x8.yml
