dataset_paths = {
	'cars_train': '',
	'cars_test': '',

	'celeba_train': '',
	'celeba_test': '',
	'celeba_test_w_inv': '',
	'celeba_test_w_latents': '',

	'ffhq': '',
	'ffhq_w_inv': '',
	'ffhq_w_latents': '',

	'afhq_wild_train': '',
	'afhq_wild_test': '',

}

model_paths = {
	# models for backbones and losses
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'resnet34': 'pretrained_models/resnet34-333f7ec4.pth',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pt',
	# stylegan2 generators
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'stylegan_cars': 'pretrained_models/stylegan2-car-config-f.pt',
	'stylegan_ada_wild': 'pretrained_models/afhqwild.pt',
	# model for face alignment
	'shape_predictor': 'pretrained_models/shape_predictor_68_face_landmarks.dat',
	# models for ID similarity computation
	'curricular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	# WEncoders for training on various domains
	'faces_w_encoder': 'pretrained_models/faces_w_encoder.pt',
	'cars_w_encoder': 'pretrained_models/cars_w_encoder.pt',
	'afhq_wild_w_encoder': 'pretrained_models/afhq_wild_w_encoder.pt',
	# models for domain adaptation
	'restyle_e4e_ffhq': 'pretrained_models/restyle_e4e_ffhq_encode.pt',
	'stylegan_pixar': 'pretrained_models/pixar.pt',
	'stylegan_toonify': 'pretrained_models/ffhq_cartoon_blended.pt',
	'stylegan_sketch': 'pretrained_models/sketch.pt',
	'stylegan_disney': 'pretrained_models/disney_princess.pt'
}

edit_paths = {
	'age': 'editing/interfacegan_directions/age.pt',
	'smile': 'editing/interfacegan_directions/smile.pt',
	'pose': 'editing/interfacegan_directions/pose.pt',
	'cars': 'editing/ganspace_directions/cars_pca.pt',
	'styleclip': {
		'delta_i_c': 'editing/styleclip/global_directions/ffhq/fs3.npy',
		's_statistics': 'editing/styleclip/global_directions/ffhq/S_mean_std',
		'templates': 'editing/styleclip/global_directions/templates.txt'
	}
}