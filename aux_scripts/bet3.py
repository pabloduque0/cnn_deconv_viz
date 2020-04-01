import subprocess
import os

file_name = "FLAIR.nii.gz"
base_command = 'hd-bet -i {}'
for root, dirs, files in os.walk('../'):
    if file_name in set(files):
        print(os.path.join(root, file_name))

        process = subprocess.Popen(base_command.format(os.path.join(root, file_name)).split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        print('OUTPUT: ', output)
        print('ERROR: ', error)
