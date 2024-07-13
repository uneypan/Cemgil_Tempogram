# Cemgil Tempogram

![](image.png)

This is a python implementation of Cemgil's tempogram.

> Cemgil A T, Kappen B, Desain P, et al. On tempo tracking: Tempogram representation and Kalman filtering[J]. Journal of New Music Research, 2000, 29(4): 259-273.


**Requirments**:
- torch
- madmom
- librosa
- numpy
- matplotlib

**Simple usage**:
```python
from cemgil.rythm import cemgil_tempogram
Tg = cemgil_tempogram('cemgil/test.wav',
                        fps=100, 
                        d_range=(1, 300), 
                        a_m=[0.25, 0.5, 1, 0.5, 0.25], 
                        device='cpu')
```
Result `Tg` shape align with [librosa.feature.tempogram](https://librosa.org/doc/latest/generated/librosa.feature.tempogram.html).

`device` set to `'cuda'` to use Torch version, otherwise Numpy.




