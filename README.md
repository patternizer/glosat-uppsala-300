![image](https://github.com/patternizer/glosat-uppsala-300/blob/main/037401-fft-smooth.png)
![image](https://github.com/patternizer/glosat-uppsala-300/blob/main/037401-seasonal.png)

# glosat-uppsala-300

Python codebase to calculate FFT-filtered seasonal means for long station timeseries such as CET. Part of ongoing work for the [GloSAT](https://www.glosat.org) project: www.glosat.org. 

## Contents

* `uppsala-300.py` - python script to compare seasonal means from Uppsala and Stockholm
* `extract-station-seasonal-means.py` - python script to calculate MA, LOESS and FFT smoothing and seasonal means for a GloSAT.p03 station
* `ts-fft-filter-lowpass-no-window.py` - python script FFT low pass filter using a cut-off frequency
* `ts-fft-filter-lowpass-hamming-window.py` - python script FFT low pass filter using a hamming window filter
* `ts-fft-filter-lowpass-extrapolation.py` - python script FFT low pass filter extrapolation
* `filter_cru.py` - python script of IDL filter_cru.pro Gaussian filter code

## Instructions for use

The first step is to clone the latest glosat-uppsala-300 code and step into the installed Github directory: 

    $ git clone https://github.com/patternizer/glosat-uppsala-300.git
    $ cd glosat-uppsala-300

Then create a DATA/ directory and copy to it the required GloSAT.p03 pickled temperature archive file.

### Using Standard Python

The code is designed to run in an environment using Miniconda3-latest-Linux-x86_64.

    $ python uppsala-300.py
    $ python extract-station-seasonal-means.py
    $ python ts-fft-filter-lowpass-no-window.py
    $ python ts-fft-filter-lowpass-hamming-window.py
    $ python ts-fft-filter-lowpass-extrapolation.py
    $ python filter_cru.py

This will generate plots of the smoothed timeseries and seasonal mean extracts.

## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)


