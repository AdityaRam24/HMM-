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

        stage('Lint & Test') {
            steps {
                bat """
                   REM Lint (if you add flake8)
                   REM "%PYTHON%" -m flake8 .
                   "%PYTHON%" -m pytest --maxfail=1 --disable-warnings -q
                """
            }
        }

        stage('Package') {
            steps {
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
        always { echo '🛠 Pipeline complete.' }
        success { echo '✅ Build succeeded!' }
        failure { echo '❌ Build failed!' }
    }
}
