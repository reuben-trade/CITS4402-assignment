from feature_extraction import FaceProcessor

fp = FaceProcessor()
fp.process_faces("faces")
groups, _ = fp.similarity()
fp.plot_clusters(groups)
