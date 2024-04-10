IMAGE_ROOT?=localhost
IMAGE=magma
IMAGE_VERSION=0.1.0


build: Dockerfile
	docker buildx build --platform linux/amd64 \
		-t $(IMAGE_ROOT)/$(IMAGE):$(IMAGE_VERSION) \
		.

push:
	docker push $(IMAGE_ROOT)/$(IMAGE):$(IMAGE_VERSION)

singularity:
	rm -f $(IMAGE).sif $(IMAGE).tar
	docker save $(IMAGE_ROOT)/$(IMAGE):$(IMAGE_VERSION) -o $(IMAGE).tar
	singularity build $(IMAGE).sif docker-archive://$(IMAGE).tar
	rm -f $(IMAGE).tar
