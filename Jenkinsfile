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
                   echo Installing dependencies…
                   "%PYTHON%" -m pip install --upgrade pip
                   "%PYTHON%" -m pip install -r requirements.txt
                   "%PYTHON%" -m pip install pytest
                """
            }
        }

        stage('Test') {
            steps {
                bat """
                   echo Running pytest…
                   "%PYTHON%" -m pytest tests --maxfail=1 --disable-warnings -q
                """
            }
        }

        stage('Build (optional)') {
            steps {
                bat 'echo Build complete!'
            }
        }
    }

    post {
        always  { echo '🛠 Pipeline finished.' }
        success { echo '✅ Build succeeded!' }
        failure { echo '❌ Build failed!' }
    }
}
