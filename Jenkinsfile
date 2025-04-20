pipeline {
    agent any

    environment {
        PYTHON = 'C:\\Users\\admin\\AppData\\Local\\Programs\\Python\\Python313\\python.exe'
    }

    stages {
        stage('Checkout') {
            steps {
                git url: 'https://github.com/AdityaRam24/HMM-.git', branch: 'main'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat """
                   "%PYTHON%" -m pip install --upgrade pip
                   "%PYTHON%" -m pip install -r requirements.txt
                """
            }
        }

        stage('Package') {
            steps {
                // if you don't have setup.py, you can skip this
                bat """
                   "%PYTHON%" -m pip install wheel
                   "%PYTHON%" setup.py bdist_wheel
                """
            }
        }

        stage('Archive Artifact') {
            steps {
                archiveArtifacts artifacts: 'dist/*.whl', fingerprint: true
            }
        }
    }

    post {
        always   { echo '🚀 Pipeline finished.' }
        success  { echo '✅ Build succeeded!' }
        failure  { echo '❌ Build failed!' }
    }
}
