"""
Integrantes:
    - Miguel Angel Toro
    - Stefania Concha Caceres
"""

import os
import pydicom
import numpy as np
import pandas as pd
import cv2


class ProcesadorDICOM:

    def __init__(self, ruta):
        self.ruta = ruta
        self.archivos = []
        self.dataframe = None

    # 1. Carga de archivos

    def cargar_archivos(self):

        encontrados = 0
        for nombre in os.listdir(self.ruta):
            ruta_archivo = os.path.join(self.ruta, nombre)
            try:
                dataset = pydicom.dcmread(ruta_archivo)
                self.archivos.append(dataset)
                encontrados += 1
                print(f"  Cargado: {nombre}")
            except Exception:
                # El archivo no es un DICOM válido, se omite
                print(f"  Omitido (no es DICOM): {nombre}")

        print(f"\n  Total cargados: {encontrados} archivo(s)\n")

    
    # 2. Extracción de metadatos

    def extraer_metadatos(self):

        def obtener_tag(dataset, tag):
            # Retorna el valor del tag o 'N/A' si no existe (anonimizado)
            return str(getattr(dataset, tag, "N/A"))

        filas = []
        for ds in self.archivos:
            fila = {
                "PatientID"          : obtener_tag(ds, "PatientID"),
                "PatientName"        : obtener_tag(ds, "PatientName"),
                "StudyInstanceUID"   : obtener_tag(ds, "StudyInstanceUID"),
                "StudyDescription"   : obtener_tag(ds, "StudyDescription"),
                "StudyDate"          : obtener_tag(ds, "StudyDate"),
                "Modality"           : obtener_tag(ds, "Modality"),
                "Rows"               : obtener_tag(ds, "Rows"),
                "Columns"            : obtener_tag(ds, "Columns"),
            }
            filas.append(fila)

        self.dataframe = pd.DataFrame(filas)
        print("  Metadatos extraídos correctamente.")
        print(self.dataframe.to_string())
        print()

    
    # 3. Análisis de intensidad con NumPy


    def analizar_intensidad(self):
        """
        Calcula el promedio de intensidad de píxeles de cada imagen
        y lo agrega como columna 'IntensidadPromedio' al DataFrame.
        """
        promedios = []
        for ds in self.archivos:
            try:
                promedio = float(np.mean(ds.pixel_array))
                promedios.append(round(promedio, 2))
            except Exception:
                # El archivo no tiene datos de píxeles (SR, PR, etc.)
                promedios.append(None)

        self.dataframe["IntensidadPromedio"] = promedios
        print("  Intensidad promedio calculada.")
        print(self.dataframe[["PatientID", "Modality", "IntensidadPromedio"]].to_string())
        print()