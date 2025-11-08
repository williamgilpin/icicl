Loop the videos
+ 


information recovery across scales
3:03


Slide 1: 
+ Weather: transfer across scales. From Large scales to small scales. (forward cascade)
+ + Mesoscale systems: energy goes from small scales to large scales. (inverse cascade)
Stock market: ??
Visual system: neurons fire in response to features, information gets combined (inverse cascade)

Slide 2: 
+ Paper crumpling: 
+ + As the paper crumples, the papers in the paper create fractal objects, self-similar over multiple scales.
+ Needs some practice for smoothness

Slide 3 (3:10): 2D turbluence

Slide 4: Navier-Stokes equations: 
+ Vorticity equation
+ Forcing term important

Slide 5: Decaying turbulence in the absence of forcing


Slide 6: Turbulent cascade
+ 


Slide 7 (3:14): Turbulent cascade 2
+ Description 


Slide 8: Home slide (3:15)
+ Spend more time on this


Slide 9: 2 gets dissipated (movie)

Home slide (3:18)

Inverting a flow field (inverse problem)
+ Describe the math in more detail. What is an inverse problem

AMI vs time decays
+ Is there an expected scaling here for a chaotic system?

FNO: Lift then predict

What does a function to function map do?
+ 

On the slide showing decryption from a filtered vorticitiy field: the low-pass filter image doesn't look right. It doesn't look like the original vorticity field.

More time/description of "technical slides"
+ Spend more time describing the inverse cascade, it's really important to convey what that graph shows and how it relates to the informal introduction. Maybe find a better version of that graph (Boffetta paper)? The one you have has a lot of annotations.
+ Fully describe the inverse problem formulation
+ 1ish additional slide on theoretical interest, maybe the data processing inequality
+ Spend more time walking us through the FNO. Consider adding a following slide describing the training setting and hyperparameters, with an emphasis on how they were chosen.

Revisit the "cascade schematic" when you describe the injection/readout: Information goes in here, and we read it out here and there

Whenever possible, show don't tell. Use pictures and movies instead of bullets whenever possible. For all slides that have 0 visuals, consider whether a diagram or visual can convey the same idea. The "spectral bias" comes to mind as an example where simply showing a picture of a biased solve (in real space or Fourier space) would convey the idea visually. For the approximation slides, maybe a diagram showing an MLP, wider is better. Not sure what the equivalent would be for operator approximation theorems.




**TODO** Technical: can probably write a measure of "gain" over the raw field rather than comparing the baselines

Data processing inequality: 

**Bullet points** replace with pictures: spectral bias? show an example
