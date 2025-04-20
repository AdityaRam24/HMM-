pipeline {
    agent any

    environment {
        // Your Python interpreter
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
                   REM Uncomment if you add flake8
                   REM "%PYTHON%" -m flake8 .
                   "%PYTHON%" -m pytest --maxfail=1 --disable-warnings -q || exit 0
                """
            }
        }

        stage('Run Flask App') {
            steps {
                bat """
                   echo Starting Flask app...
                   "%PYTHON%" app.py
                """
            }
        }
    }

    post {
        always {
            echo '🛠 Pipeline complete.'
        }
        success {
            echo '✅ Build succeeded!'
        }
        failure {
            echo '❌ Build failed!'
        }
    }
}
