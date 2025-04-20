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
                script {
                    if (isUnix()) {
                        // Linux/macOS: create and activate venv
                        sh 'python3 -m venv "$VENV_HOME"'
                        sh '. "$VENV_HOME/bin/activate"'
                    } else {
                        // Windows: create and activate venv
                        bat '"C:\\Users\\admin\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" -m venv "%VENV_HOME%"'
                        bat 'call "%VENV_HOME%\\Scripts\\activate.bat"'
                    }
                }
            }
        }
        stage('Install Dependencies') {
            steps {
                script {
                    if (isUnix()) {
                        // Upgrade pip and install dependencies on Unix
                        sh '. "$VENV_HOME/bin/activate" && pip install --upgrade pip'
                        sh '. "$VENV_HOME/bin/activate" && pip install -r requirements.txt'
                    } else {
                        // Use python -m pip on Windows to upgrade pip
                        bat 'call "%VENV_HOME%\\Scripts\\activate.bat" && python -m pip install --upgrade pip'
                        bat 'call "%VENV_HOME%\\Scripts\\activate.bat" && pip install -r requirements.txt'
                    }
                }
            }
        }
        stage('Lint') {
            steps {
                script {
                    if (isUnix()) {
                        sh '. "$VENV_HOME/bin/activate" && pip install flake8'
                        sh '. "$VENV_HOME/bin/activate" && flake8 .'
                    } else {
                        bat 'call "%VENV_HOME%\\Scripts\\activate.bat" && pip install flake8'
                        bat 'call "%VENV_HOME%\\Scripts\\activate.bat" && flake8 .'
                    }
                }
            }
        }
        stage('Test') {
            steps {
                script {
                    if (isUnix()) {
                        sh '. "$VENV_HOME/bin/activate" && pytest --maxfail=1 --disable-warnings -q'
                    } else {
                        bat 'call "%VENV_HOME%\\Scripts\\activate.bat" && pytest --maxfail=1 --disable-warnings -q'
                    }
                }
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
                script {
                    // Push to Docker Hub; requires a Jenkins credential with ID 'dockerhub-credentials'
                    withCredentials([usernamePassword(credentialsId: 'dockerhub-credentials', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                        if (isUnix()) {
                            sh 'echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin'
                            sh "docker push adityaram24/hmm-app:${env.BUILD_NUMBER}"
                        } else {
                            bat 'echo %DOCKER_PASS% | docker login -u %DOCKER_USER% --password-stdin'
                            bat 'docker push adityaram24/hmm-app:%BUILD_NUMBER%'
                        }
                    }
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
