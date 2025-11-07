@echo off
echo IDDAW Project Results Generator
echo ==============================
echo.

cd /d "%~dp0.."
set OUTPUT_DIR=project\outputs\report
set SPLITS_DIR=project\splits

echo Creating output directory...
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo.
echo Step 1: Running model evaluations...
echo -----------------------------------

echo Running RGB MobileNetV3 evaluation...
python -m project.eval --splits_dir %SPLITS_DIR% --ckpt project\ckpts\best_rgb_mbv3.pt --modality rgb --backbone mbv3 --image_size 512 --batch_size 4 --workers 4 > "%OUTPUT_DIR%\rgb_mbv3_results.txt"

echo Running NIR FastSCNN evaluation...
python -m project.eval --splits_dir %SPLITS_DIR% --ckpt project\ckpts\best_nir_fastscnn.pt --modality nir --backbone fastscnn --image_size 512 --batch_size 4 --workers 4 > "%OUTPUT_DIR%\nir_fastscnn_results.txt"

echo Running Early Fusion evaluation...
python -m project.eval --splits_dir %SPLITS_DIR% --ckpt project\ckpts\best_early4_mbv3.pt --modality early4 --backbone mbv3 --image_size 512 --batch_size 4 --workers 4 > "%OUTPUT_DIR%\early4_mbv3_results.txt"

echo Running Mid Fusion evaluation...
python -m project.eval --splits_dir %SPLITS_DIR% --ckpt project\ckpts\best_mid_mbv3.pt --modality mid --backbone mbv3 --image_size 512 --batch_size 4 --workers 4 > "%OUTPUT_DIR%\mid_mbv3_results.txt"

echo.
echo Step 2: Generating visualizations...
echo ----------------------------------

echo Generating RGB MobileNetV3 visualizations...
python -m project.demo --splits_dir %SPLITS_DIR% --ckpt project\ckpts\best_rgb_mbv3.pt --modality rgb --backbone mbv3 --num_samples 20 --out_dir "%OUTPUT_DIR%\rgb_mbv3"

echo Generating NIR FastSCNN visualizations...
python -m project.demo --splits_dir %SPLITS_DIR% --ckpt project\ckpts\best_nir_fastscnn.pt --modality nir --backbone fastscnn --num_samples 20 --out_dir "%OUTPUT_DIR%\nir_fastscnn"

echo Generating Early Fusion visualizations...
python -m project.demo --splits_dir %SPLITS_DIR% --ckpt project\ckpts\best_early4_mbv3.pt --modality early4 --backbone mbv3 --num_samples 20 --out_dir "%OUTPUT_DIR%\early4_mbv3"

echo Generating Mid Fusion visualizations...
python -m project.demo --splits_dir %SPLITS_DIR% --ckpt project\ckpts\best_mid_mbv3.pt --modality mid --backbone mbv3 --num_samples 20 --out_dir "%OUTPUT_DIR%\mid_mbv3"

echo.
echo Step 3: Generating advanced visualizations and graphs...
echo ----------------------------------------------------

echo Running visualization script to generate graphs and charts...
python scripts\generate_visualizations.py --results_dir "%OUTPUT_DIR%" --output_dir "%OUTPUT_DIR%\visualizations"

echo.
echo Step 4: Compiling results summary...
echo ----------------------------------

echo Compiling results into summary file...
echo # IDDAW Project Results Summary > "%OUTPUT_DIR%\results_summary.md"
echo Generated on %date% at %time% >> "%OUTPUT_DIR%\results_summary.md"
echo. >> "%OUTPUT_DIR%\results_summary.md"
echo ## Model Performance Metrics >> "%OUTPUT_DIR%\results_summary.md"
echo. >> "%OUTPUT_DIR%\results_summary.md"
echo ### RGB MobileNetV3 >> "%OUTPUT_DIR%\results_summary.md"
type "%OUTPUT_DIR%\rgb_mbv3_results.txt" >> "%OUTPUT_DIR%\results_summary.md"
echo. >> "%OUTPUT_DIR%\results_summary.md"
echo ### NIR FastSCNN >> "%OUTPUT_DIR%\results_summary.md"
type "%OUTPUT_DIR%\nir_fastscnn_results.txt" >> "%OUTPUT_DIR%\results_summary.md"
echo. >> "%OUTPUT_DIR%\results_summary.md"
echo ### Early Fusion MobileNetV3 >> "%OUTPUT_DIR%\results_summary.md"
type "%OUTPUT_DIR%\early4_mbv3_results.txt" >> "%OUTPUT_DIR%\results_summary.md"
echo. >> "%OUTPUT_DIR%\results_summary.md"
echo ### Mid Fusion MobileNetV3 >> "%OUTPUT_DIR%\results_summary.md"
type "%OUTPUT_DIR%\mid_mbv3_results.txt" >> "%OUTPUT_DIR%\results_summary.md"
echo. >> "%OUTPUT_DIR%\results_summary.md"

echo ## Visualization Outputs >> "%OUTPUT_DIR%\results_summary.md"
echo. >> "%OUTPUT_DIR%\results_summary.md"
echo ### Model Prediction Samples >> "%OUTPUT_DIR%\results_summary.md"
echo. >> "%OUTPUT_DIR%\results_summary.md"
echo The following visualization directories have been generated: >> "%OUTPUT_DIR%\results_summary.md"
echo. >> "%OUTPUT_DIR%\results_summary.md"
echo - RGB MobileNetV3: `%OUTPUT_DIR%\rgb_mbv3\` >> "%OUTPUT_DIR%\results_summary.md"
echo - NIR FastSCNN: `%OUTPUT_DIR%\nir_fastscnn\` >> "%OUTPUT_DIR%\results_summary.md"
echo - Early Fusion MobileNetV3: `%OUTPUT_DIR%\early4_mbv3\` >> "%OUTPUT_DIR%\results_summary.md"
echo - Mid Fusion MobileNetV3: `%OUTPUT_DIR%\mid_mbv3\` >> "%OUTPUT_DIR%\results_summary.md"
echo. >> "%OUTPUT_DIR%\results_summary.md"
echo Each directory contains the following visualization types: >> "%OUTPUT_DIR%\results_summary.md"
echo. >> "%OUTPUT_DIR%\results_summary.md"
echo - Original images >> "%OUTPUT_DIR%\results_summary.md"
echo - Segmentation masks >> "%OUTPUT_DIR%\results_summary.md"
echo. >> "%OUTPUT_DIR%\results_summary.md"
echo ### Performance Visualizations and Graphs >> "%OUTPUT_DIR%\results_summary.md"
echo. >> "%OUTPUT_DIR%\results_summary.md"
echo Advanced visualizations have been generated in the `%OUTPUT_DIR%\visualizations\` directory: >> "%OUTPUT_DIR%\results_summary.md"
echo. >> "%OUTPUT_DIR%\results_summary.md"
echo - Performance comparison bar charts >> "%OUTPUT_DIR%\results_summary.md"
echo - Model performance radar charts >> "%OUTPUT_DIR%\results_summary.md"
echo - Confusion matrices >> "%OUTPUT_DIR%\results_summary.md"
echo - Sample prediction grid visualizations >> "%OUTPUT_DIR%\results_summary.md"
echo. >> "%OUTPUT_DIR%\results_summary.md"
echo An interactive HTML report with all visualizations is available at: `%OUTPUT_DIR%\visualizations\report.html` >> "%OUTPUT_DIR%\results_summary.md"
echo - Overlay visualizations >> "%OUTPUT_DIR%\results_summary.md"
echo - Confidence/entropy maps >> "%OUTPUT_DIR%\results_summary.md"
echo - Safety visualizations >> "%OUTPUT_DIR%\results_summary.md"

echo.
echo Results generation complete!
echo All results have been saved to: %OUTPUT_DIR%
echo Summary file: %OUTPUT_DIR%\results_summary.md
echo.
echo You can now use these results in your report.
echo.