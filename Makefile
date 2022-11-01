COMPOSE_CMD	= docker compose
USER_ID		= $(shell id -u)
USER_NAME	= $(shell id -u --name)
GROUP_ID	= $(shell id -u)
DOCKER_DIR	= docker/
BUILD_ARGS	= --build-arg USER_ID=$(USER_ID) --build-arg USER_NAME=$(USER_NAME) --build-arg GROUP_ID=$(GROUP_ID)


run:
	$(COMPOSE_CMD) -f $(DOCKER_DIR)docker-compose-experiments.yml up

run-jupyter:
	$(COMPOSE_CMD) -f $(DOCKER_DIR)docker-compose-jupyter.yml up

run-maggie:
	$(COMPOSE_CMD) -f $(DOCKER_DIR)docker-compose-maggie.yml up

build:
	mv pretrained_models ../.
	$(COMPOSE_CMD) -f $(DOCKER_DIR)docker-compose-experiments.yml build $(BUILD_ARGS)
	mv ../pretrained_models .

build-jupyter:
	mv pretrained_models ../.
	$(COMPOSE_CMD) -f $(DOCKER_DIR)docker-compose-jupyter.yml build $(BUILD_ARGS)
	mv ../pretrained_models .

build-maggie:
	mv pretrained_models ../.
	$(COMPOSE_CMD) -f $(DOCKER_DIR)docker-compose-maggie.yml build $(BUILD_ARGS)
	mv ../pretrained_models .

down:
	$(COMPOSE_CMD) -f $(DOCKER_DIR)docker-compose-jupyter.yml down

down-maggie:
	$(COMPOSE_CMD) -f $(DOCKER_DIR)docker-compose-maggie.yml down

clean:
	rm -rf pretrained_models/**/training-logs/checkpoints/

