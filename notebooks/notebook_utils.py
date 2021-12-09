import os

HYPERSTYLE_PATHS = {
    "faces": {"id": "1C3dEIIH1y8w1-zQMCyx7rDF0ndswSXh4", "name": "hyperstyle_ffhq.pt"},
    "cars": {"id": "1WZ7iNv5ENmxXFn6dzPeue1jQGNp6Nr9d", "name": "hyperstyle_cars.pt"},
    "afhq_wild": {"id": "1OMAKYRp3T6wzGr0s3887rQK-5XHlJ2gp", "name": "hyperstyle_afhq_wild.pt"}
}
W_ENCODERS_PATHS = {
    "faces": {"id": "1M-hsL3W_cJKs77xM1mwq2e9-J0_m7rHP", "name": "faces_w_encoder.pt"},
    "cars": {"id": "1GZke8pfXMSZM9mfT-AbP1Csyddf5fas7", "name": "cars_w_encoder.pt"},
    "afhq_wild": {"id": "1MhEHGgkTpnTanIwuHYv46i6MJeet2Nlr", "name": "afhq_wild_w_encoder.pt"}
}
FINETUNED_MODELS = {
    "toonify": {'id': '1r3XVCt_WYUKFZFxhNH-xO2dTtF6B5szu', 'name': 'toonify.pt'},
    "pixar": {'id': '1trPW-To9L63x5gaXrbAIPkOU0q9f_h05', 'name': 'pixar.pt'},
    "sketch": {'id': '1aHhzmxT7eD90txAN93zCl8o9CUVbMFnD', 'name': 'sketch.pt'},
    "disney_princess": {'id': '1rXHZu4Vd0l_KCiCxGbwL9Xtka7n3S2NB', 'name': 'disney_princess.pt'}
}
RESTYLE_E4E_MODELS = {'id': '1e2oXVeBPXMQoUoC_4TNwAWpOPpSEhE_e', 'name': 'restlye_e4e.pt'}


class Downloader:

    def __init__(self, code_dir, use_pydrive):
        self.use_pydrive = use_pydrive
        current_directory = os.getcwd()
        self.save_dir = os.path.join(os.path.dirname(current_directory), code_dir, "pretrained_models")
        os.makedirs(self.save_dir, exist_ok=True)
        if self.use_pydrive:
            self.authenticate()

    def authenticate(self):
        from google.colab import auth
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
        from oauth2client.client import GoogleCredentials

        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        self.drive = GoogleDrive(gauth)

    def download_file(self, file_id, file_name):
        file_dst = f'{self.save_dir}/{file_name}'
        if os.path.exists(file_dst):
            print(f'{file_name} already exists!')
            return
        if self.use_pydrive:
            downloaded = self.drive.CreateFile({'id':file_id})
            downloaded.FetchMetadata(fetch_all=True)
            downloaded.GetContentFile(file_dst)
        else:
            os.system(f"gdown --id {file_id} -O {file_dst}")


def run_alignment(image_path):
    import dlib
    from scripts.align_faces_parallel import align_face
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print('Downloading files for aligning face image...')
        os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
        print('Done.')
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print(f"Finished running alignment on image: {image_path}")
    return aligned_image
