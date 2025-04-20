pipeline {
    agent any
    environment {
        VENV_HOME = "${WORKSPACE}/venv"
    }
    stages {
        stage('Checkout') {
            steps {
                // Pull the latest code from the Git repository
                checkout scm
            }
        }
        stage('Setup Python') {
            steps {
                // Create and activate a virtual environment
                sh 'python3 -m venv "$VENV_HOME"'
                sh '. "$VENV_HOME/bin/activate"'
            }
        }
        stage('Install Dependencies') {
            steps {
                // Upgrade pip and install project dependencies
                sh '. "$VENV_HOME/bin/activate" && pip install --upgrade pip'
                sh '. "$VENV_HOME/bin/activate" && pip install -r requirements.txt'
            }
        }
        stage('Lint') {
            steps {
                // Optional: install and run a linter
                sh '. "$VENV_HOME/bin/activate" && pip install flake8'
                sh '. "$VENV_HOME/bin/activate" && flake8 .'
            }
        }
        stage('Test') {
            steps {
                // Run your test suite (if any)
                sh '. "$VENV_HOME/bin/activate" && pytest --maxfail=1 --disable-warnings -q'
            }
        }
        stage('Build Docker Image') {
            when {
                expression { fileExists 'Dockerfile' }
            }
            steps {
                script {
                    // Build Docker image with tag based on build number
                    dockerImage = docker.build("adityaram24/hmm-app:${env.BUILD_NUMBER}")
                }
            }
        }
        stage('Push Docker Image') {
            when {
                expression { fileExists 'Dockerfile' }
            }
            steps {
                // Push to Docker Hub; requires a Jenkins credential with ID 'dockerhub-credentials'
                withCredentials([usernamePassword(credentialsId: 'dockerhub-credentials', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    sh 'echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin'
                    sh "docker push adityaram24/hmm-app:${env.BUILD_NUMBER}"
                }
            }
        }
    }
    post {
        always {
            // Clean up workspace after build
            cleanWs()
        }
    }
}
