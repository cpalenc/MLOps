# Project 1 - Grupo 1

![alt text](https://github.com/oecorrechag/MLOps/blob/main/images/logo.PNG)

Class: MLOps <br>
Code: 11179 <br>
Professor: Cristian Diaz Alvarez <br>
Members:

    Daniel Chavarro - @anielFchavarro
    Cristhian Palencia - @cpalenc
    Oscar Correa - @oecorrechag

## Table of Contents

- [Usar con Docker](#Docker)
- [Usar con Google colab](#Google)

### Docker

How to run docker compose

1. Download the repository.

```bash
ssh git clone https://github.com/cpalenc/MLOps.git
```

2. Navigate to the PROYECTO_1 branch within the project_1 folder.

```bash
cd MLOps/
git checkout proyecto_1
cd proyecto_1
```

3. Run the following Docker command:

```bash
docker compose up
```

4. This command will display the following:

![alt text](https://github.com/oecorrechag/MLOps/blob/main/images/open_html.png)

In the image, the console output is displayed, and the box indicates the token. This token should be entered into an internet browser (of your choice) to launch the Jupyter Notebook.

5. Once inside Jupyter, click on the "Build" option to access the content of Project 1. 

![alt text](https://github.com/oecorrechag/MLOps/blob/main/images/open_jupyter.png)

6. Open the file TFT_convertype.ipynb.

Within this file, you will find the solution to the project. 

![alt text](https://github.com/oecorrechag/MLOps/blob/main/images/open_project.PNG)

### Google

There is an alternative to using the document using Google Colab. To utilize it, please follow these steps:

![alt text](https://github.com/oecorrechag/MLOps/blob/main/images/google.PNG)

1. Upload the TFT_convertype.ipynb file into a Google Colab Notebook.

2. Add the additional modules (listed below) in the files section that are necessary to run the TFT_convertype.ipynb Notebook.

    - cover_type_constans.py
    - cover_type_transform.py

3. Add the following code statement before the "librerias" cell.

```bash
!pip install tfx
```
Nota: Installing this library may take between 5 to 10 minutes.

4. It is possible that during the first execution, it may request a kernel restart; if so, the kernel will restart automatically.

![alt text](https://github.com/oecorrechag/MLOps/blob/main/images/google_kernel.PNG)

<hr>

[Go to Top](#Table-of-Contents)
