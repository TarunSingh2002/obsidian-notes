---
tags:
  - MLOps
  - docker
  - containers
---
# Docker

## How Docker Works
![[Docker - working.png]]

### Docker Engine
- Docker Engine is the core component of the Docker platform. It creates, runs, and manages Docker containers, and is the runtime that powers Docker's containerization capabilities
- ![[Docker - engine.png]]

### Docker Image
- A Docker image is a lightweight, stand-alone, executable software package that includes everything needed to run a piece of software: the code, runtime, libraries, environment variables, and configuration files
- Images are used to create Docker containers, which are instances of these images

### Docker Container
- A Docker container is a lightweight, portable, isolated environment that wraps an application and its dependencies, so it runs the same way across different computing environments
- Containers are created from Docker images, which are immutable and contain everything the application needs to run
- Each container has its own isolated OS environment (filesystem, processes), but all containers share the host machine's kernel

### Docker Compose
- Used to create multiple containers and manage them all at once

---

## Commands

### Image
- Show all images
	```bash
	docker images
	```
- Pull an image from Docker Hub
	```bash
	docker pull <image-name>
	```
- Build a Docker image from a Dockerfile
	```bash
	docker build -t userName/ImageName .
	```
- Push the image to Docker Hub
	```bash
	docker push username/<image-name>
	```

### Container
- Show all Docker containers
	```bash
	docker container ls -a
	```
	- Only the running ones
		```bash
		docker container ls
		```
- Stop a container
	```bash
	docker stop <container_name>
	```

#### Run a Docker image
- **From an image**
	- With port mapping and interactive mode
		```bash
		docker run -it --name <name> -p 8008:5000 <image-name>
		```
		- Port mapping is `-p <host port>:<container port>`
			- `8008` is the port on your host machine (your computer) where you want to access the application
			- `5000` is the port mentioned in the Dockerfile, where the app is actually running inside the container
	- Running Ubuntu
		```bash
		docker run -it ubuntu
		```
		- `-it` means run the container in interactive mode
	- Note: if the Docker daemon doesn't find the image on your machine, it searches Docker Hub, pulls it, and then runs it
- **From an existing container**
	- Start the container first
		```bash
		docker start <container_name>
		```
	- Go inside the container in interactive mode
		```bash
		docker exec -it <container_name> bash
		```
	- Exit with `Ctrl + D`

### Compose
- Start the containers
	```bash
	sudo docker compose up
	```
- Stop the containers
	```bash
	docker compose down
	```

---

## Dockerfile
- Contains 6 things
	1. Base image
	2. Create a working directory
	3. Copy the code to the working directory
	4. Install the requirements file
	5. Port that needs to be exposed
	6. Final run command
- Note: inside the container your app must listen on `0.0.0.0`, not `localhost`, or it won't be reachable from outside the container
- Example
	```dockerfile
	FROM python:3.9

	WORKDIR /app

	COPY . /app

	RUN pip install -r requirements.txt

	EXPOSE 8501

	CMD ["streamlit", "run", "app.py"]
	```

---

## Docker Compose
- File name: `docker-compose.yml`
	```yaml
	version: '3.9'

	services:
	  postgres:
	    image: postgres   # hub.docker.com
	    ports:
	      - "5432:5432"
	    environment:
	      POSTGRES_USER: postgres
	      POSTGRES_DB: review
	      POSTGRES_PASSWORD: password
	  redis:
	    image: redis
	    ports:
	      - "6379:6379"
	```
- Command to run Docker Compose (in the background)
	```bash
	docker compose up -d
	```
