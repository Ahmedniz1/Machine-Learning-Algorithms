from src.utils import check_random_state
def make_blobs(n_samples= 100 ,n_features= 2, centers = 3, cluster_std = 1.0, center_box=(-10.0, 10.0), random_state = None):
    print('haha')
    generator  = check_random_state(random_state)
    n_centers = centers
    centers = generator.uniform(center_box[0],center_box[1],size = (centers,n_features))
    
