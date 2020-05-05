
This is an implementation for vision and language multimodal research developed on top of Pythia 

<a href="https://readthedocs.org/projects/learnpythia/">
    <img width="20%" height="1%" alt="Pythia" src="https://i.imgur.com/wPgp4N4.png"/>
</a>




### Model Implementation

We have proposed few major improvements over Pythia, which is an implementation for the <a href="https://arxiv.org/pdf/1904.08920.pdf">LoRRA model<a>. The dynamic answering space is expanded by adding an Instance segment module. We have also replaced the existing OCR with spell correcting OCR and add the spatial features of the OCR. We have also implemented n-gram to the modified OCR.



<img width="80%" height="40%" alt="Pythia" src="https://imgur.com/rv4694M.png"/>




### Test

For testing run our <a href = "https://colab.research.google.com/drive/1wVRPtYPO4pdMljFcqFxvQy43WcIZQrGJ">notebook<a>






