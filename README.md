# Matchnet-For-VOT

https://github.com/hanxf/matchnet adapted for VOT dataset

Files have been edited to run without .sh files. This is for the purpose of testing only. Python files of:

    1. generate_patch_dp.py: run this to generate yosemite dataset

    - to reduce time to experiment, run for only one dataset.
    - Procedure to run for other datasets can be found here: https://github.com/hanxf/matchnet
    - Trained models can be downloaded from here: https://github.com/hanxf/matchnet/tree/master/models

    2. evaluate_matchnet.py: run this to train and test for yosemite only.

    - the model works well for grayscale data
    - key take aways would be: Matching is invariant to illuminations (not scale)

    3. eval_matchnetVOT2016.py: run this to compare current VOT 2016 dataset current frame with the previous frame.

    - This works fine too. Data templates are choses randomly.
    - The above point adds to the error in detection; in case templates are not choses around the grouth truth object points.
    - would work better if RGB - all 3 color channels were considered. Imples network has to be modified.

    4. eval_VOTFirst Frame.py: run this to compare current VOT 2016 dataset current frame with the first frame.

    - question to be answered: how does the matchnet compare to object variations?
    - problem: matchnet is not trained for objects, but for tetural data.
    - solution: train on object data, then look further into how network adapts to object variations.

results: Top Match and Matches above 0.95 score (brighter red implies higher score)

![00000161](https://user-images.githubusercontent.com/11435669/30651982-284f4dc8-9df5-11e7-838e-7c8c3a7245b7.jpg)
![top00000161](https://user-images.githubusercontent.com/11435669/30652004-3410b516-9df5-11e7-857a-f8fdf6ae6472.jpg)

Purpose

Testing Matchnet (courtesy - Xufeng Han)
