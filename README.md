![image](https://github.com/patternizer/glosat-uppsala-300/blob/main/uppsala-stockholm-diff.png)
![image](https://github.com/patternizer/glosat-uppsala-300/blob/main/uppsala-and-stockholm-fft-smooth.png)
![image](https://github.com/patternizer/glosat-uppsala-300/blob/main/uppsala-seasonal.png)
![image](https://github.com/patternizer/glosat-uppsala-300/blob/main/stockholm-seasonal.png)

# glosat-uppsala-300

Python codebase to extract seasonal means for Uppsala (024581) and Stockholm (024851) long ~300 yr instrumental land surface temperature timeseries. Part of ongoing work for the [GloSAT](https://www.glosat.org) project: www.glosat.org. 

## Contents

* `uppsala-300.py` - python script to compare instrumental land surface temperature timeseries from Uppsala and Stockholm
* `extract-station-seasonal-means.py` - python script to extract seasonal means

## Instructions for use

The first step is to clone the latest glosat-uppsala-300 code and step into the installed Github directory: 

    $ git clone https://github.com/patternizer/glosat-uppsala-300.git
    $ cd glosat-uppsala-300

Then create a DATA/ directory and copy to it the required GloSAT.p03 pickled temperature archive file.

### Using Standard Python

The code is designed to run in an environment using Miniconda3-latest-Linux-x86_64.

    $ python uppsala-300.py
    $ python extract-station-seasonal-means.py

This will generate plots of the smoothed timeseries and seasonal mean extracts.

## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)


