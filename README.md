# A simple API for OCR

**Guillaume Desfoges, Théo Viel**

## Introduction 

### About this project

This project is a school project at the ENPC IMI Mathematics and Computer Sciences department.
It was done by Guillaume Desforges and Théo Viel, with the supervision of Mathieu Aubry.
The main repository for this project is : https://github.com/GuillaumeDesforges/ocr-enpc

### Objectives

Our objective is to devellop an easy to use API for Optical Character Recognition.
An interesting thing in this project was how we used `keras.backend.ctc_batch_cost` and `keras.backend.ctc_decode`. Their use is not trivial and looking at our code may help quite a lot if you have no idea where to start !

### Main library used :
* Keras for the neural networks
* Tensorflow as backend
* Pyqt for the GUI

