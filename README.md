data.py and 3D_S_Coherence.py are scripts for post-processsing and analysis of the simulation data. ColdMagnetizedPlasma.py is the main simulation script. The original scripts were written by HarryGX0331 on GitHub.

If you could focus on modifying ColdMagnetizedPlasma.py, that would be amazing!

I made a few modifications to the main() function as I was running into an Out of Memory Error. Original scripts can be found at https://github.com/HarryGX0331/GPP-LAPD. The original ReadMe is also in this repo, but the usage for 3D_S.py (almost the same as ColdMagnetizedPlasma.py except for the main() function) that it mentions is different. I have added the batch script I used for ColdMagnetizedPlasma.py into this repo.

The original ReadMe (titled OldREADME.md) has more information regarding the parameters. The only additional information is that r_left determines the size of the annulus.

As for the post processing scripts, some changes regarding output folders must be made, but I'll mark those spots with comments.
