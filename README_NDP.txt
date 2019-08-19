[Sampling video]
-* python3 sample_frames.py <video file> --num 10
- Go into outputs/[video filename]_sample_frames and use the best frame

[Getting custom tank polygon and skyline]
-* python3 poly_plotter.py <path to best frame>
- plot tank shape (rmb to plot the corners too)
- copy the example cfg file: `cp ndp.cfg woodlands.cfg`
- copy the printed output into the POLYGONS section of the cfg file to make the custom polygon
- use the same python script to find out the skyline (y coordinate)
- add this y coordinate value into the SKYLINE section of the cfg file
*- python3 crowdcount_video.py <video file> --cfg <cfg file for this cam> --checkpoly
- check that the tank polygon and sky polygon are drawn properly and end once checked

[Running crowd counting]
- Go into the config file and switch off display, video output and switch on csv output
*- python3 crowdcount_video.py <video file> --cfg <cfg file for this cam>  
- this intermediate output will be stored at outputs/<video filename>_output_CSVs/<video filename>_intermediate.csv

[Post-processing]
- Make sure speed profiles wanted are set in the cfg files
*- python3 postproc.py outputs/<video filename>_output_CSVs/<video filename>_intermediate.csv --cfg <cfg file for this cam> 
- final output csvs will be generated at outputs/<video filename>_output_CSVs/, appended with the respective speed profiles





python3 crowdcount_video.py <video file> --cfg punggol.cfg --checkpoly

python3 crowdcount_video.py <video file> --cfg punggol.cfg

python3 postproc.py <intermediate.csv> --cfg punggol.cfg