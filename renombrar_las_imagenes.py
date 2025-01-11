import os

# Carpeta de salida donde están las imágenes recortadas
output_folder = r"D:\SEMESTRE7\Dedos\Fotos\Validacion\Letra_U"

# Obtener la lista de archivos en la carpeta de salida
files = os.listdir(output_folder)

# Renombrar los archivos con el formato Dedos_n
for i, filename in enumerate(files):
    # Obtener la extensión del archivo (por ejemplo, .jpg o .png)
    file_extension = os.path.splitext(filename)[1]
    
    # Crear el nuevo nombre para la imagen
    new_name = f"Dedos_{i}{file_extension}"
    
    # Renombrar el archivo
    old_path = os.path.join(output_folder, filename)
    new_path = os.path.join(output_folder, new_name)
    os.rename(old_path, new_path)

print("Renombrado completado.")
