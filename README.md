# P05
Repositorio para el código y archivos del proyecto

## Comando necesarios para funcionamiento del proyecto en local.
### **Se debe importar el entorno con las librerias instaladas con el siguiente comando.**
conda env create -f entorno.yml
(Donde se instalara un nuevo entorno con todo lo necesario)
## En caso de que no funcione.
### Se debe instalar Python 3.11:
Por incompatibilidad del modelo de red neuronal de tensorflow
#### Se debe hacer este comando para instalar las librerias necesarias de Python
pip install librosa numpy matplotlib tensorflow
## Ejecución en local
### Web
Primero se debe importar el entorno con el comando anterior y en la carpeta del github debes ejecutar "streamlit run app.py" 
### Terminal
Debes elegir entre los distintos modelos hechos (aunque recomiendo el de clasificador_hibrido/Prueba/clasificador_hibrido_MIO(84)_PRUEBA_DEFINITIVO.py) y ejecutar el siguiente comando "python [Modelo] [Cancion a examinar]"
## BIBLIOGRAFIA
Este proyecto academico ha sido hecho mediante distintas IAs y con la guia momentanea de este libro:[Python Artificial Intelligence Projects for Beginners](https://learning.oreilly.com/library/view/python-artificial-intelligence/9781789539462/?sso_link=yes&sso_link_from=universitat-politecnica-de-valencia)
API youtube-audio: [Github de alperensumeroglu](https://github.com/alperensumeroglu/yt-audio-api.git)
