pipeline {
    agent any

    environment {
        // Define Docker image name
        DOCKER_IMAGE = 'hmm-app'
        DOCKER_TAG = "${BUILD_NUMBER}"
        CONTAINER_NAME = 'hmm-app-container'
        HOST_PORT = '9090'  // Using port 9090 which is less likely to be in use
    }

    stages {
        stage('Checkout') {
            steps {
                git url: 'https://github.com/AdityaRam24/HMM-.git', branch: 'main'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat """
                    echo Building Docker image...
                    docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} .
                """
            }
        }

        stage('Test in Docker') {
            steps {
                bat """
                    echo Running tests in Docker container...
                    docker run --rm ${DOCKER_IMAGE}:${DOCKER_TAG} python -m pytest tests --maxfail=1 --disable-warnings -q || exit 1
                """
            }
        }
        
        stage('Deploy Container') {
            steps {
                bat """
                    echo Stopping any existing container...
                    docker stop ${CONTAINER_NAME} || echo "No container to stop"
                    docker rm ${CONTAINER_NAME} || echo "No container to remove"
                    
                    echo Deploying application container...
                    docker run -d --name ${CONTAINER_NAME} -p ${HOST_PORT}:5000 ${DOCKER_IMAGE}:${DOCKER_TAG}
                    
                    echo Container deployed at http://localhost:${HOST_PORT}
                    
                    timeout /t 5 /nobreak > nul
                    
                    echo Checking if container is running...
                    docker ps | findstr ${CONTAINER_NAME} || (echo "Container failed to start" && exit 1)
                """
            }
        }
    }

    post {
        failure {
            echo '❌ Build failed!'
            bat """
                echo Cleaning up resources on failure...
                docker stop ${CONTAINER_NAME} || echo "No container to stop"
                docker rm ${CONTAINER_NAME} || echo "No container to remove"
                docker rmi ${DOCKER_IMAGE}:${DOCKER_TAG} || echo "No image to remove"
            """
        }
        success {
            echo '✅ Build succeeded! Container is running at http://localhost:${HOST_PORT}'
        }
        always {
            echo '🛠 Pipeline finished.'
        }
    }
}
