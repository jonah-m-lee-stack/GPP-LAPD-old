Script Info:

ColdMagnetizedPlasma.py is the main simulation script. data.py and 3D_S_Coherence.py are scripts for post-processsing and analysis of the simulation data. The original scripts were written by HarryGX0331 on GitHub.

_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

Main Script to Modify:

ColdMagnetizedPlasma.py is the main simulation script. It currently utilizes Dedalus3 and solves a set of equations related to plasma dynamics in cylindrical coordinates. However, there is an annulus, which affects the accuracy of results. If you could focus on modifying ColdMagnetizedPlasma.py, that would be amazing!

_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

Information Regarding Simulation/Post-Processing:

The original ReadMe (titled OldREADME.md) has more information regarding the parameters. The only additional information is that r_left determines the radius of the annulus, and r_right determines the radius of the whole cylinder.
 
Original scripts can be found at https://github.com/HarryGX0331/GPP-LAPD. The original ReadMe (titled OldREADME.md) is also in this repo, but the usage for 3D_S.py (almost the same as ColdMagnetizedPlasma.py except for the main() function) that it mentions is different. I have added the batch script I used for ColdMagnetizedPlasma.py into this repo. (run_ColdMagnetizedPlasma_example.sh)

As for the post processing scripts, some changes regarding output folders must be made, but I'll mark those spots with comments. I have added example batch scripts for those as well. (data_sim.sh, run_Coherence.sh)
