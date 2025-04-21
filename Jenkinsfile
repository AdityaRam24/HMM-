pipeline {
    agent any

    environment {
        // Define Docker image name
        DOCKER_IMAGE = 'hmm-app'
        DOCKER_TAG = "${BUILD_NUMBER}"
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
                    docker run --rm ${DOCKER_IMAGE}:${DOCKER_TAG} python -m pytest tests --maxfail=1 --disable-warnings -q
                """
            }
        }

        stage('Clean Up') {
            steps {
                bat """
                    echo Cleaning up resources...
                    docker rmi ${DOCKER_IMAGE}:${DOCKER_TAG} || echo "Image removal failed but continuing"
                """
            }
        }
    }

    post {
        always  { echo '🛠 Pipeline finished.' }
        success { echo '✅ Build succeeded!' }
        failure { echo '❌ Build failed!' }
    }
}
