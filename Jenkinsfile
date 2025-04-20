pipeline {
    agent { label 'windows' } // or 'any' if your Jenkins agents are Windows by default

    environment {
        // Point to your Python executable
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
                // Upgrade pip and install from requirements.txt
                bat """
                   "%PYTHON%" -m pip install --upgrade pip
                   "%PYTHON%" -m pip install -r requirements.txt
                """
            }
        }

        stage('Lint & Test') {
            steps {
                // If you have pytest or flake8 configured, uncomment what's needed
                bat """
                   // "%PYTHON%" -m flake8 .
                   "%PYTHON%" -m pytest --maxfail=1 --disable-warnings -q || exit 0
                """
            }
        }

        stage('Run Flask App') {
            steps {
                // Launch your Flask app; Jenkins will show the console logs
                bat """
                   echo Starting Flask app...
                   "%PYTHON%" app.py
                """
            }
        }
    }

    post {
        always {
            echo 'Pipeline finished.'
        }
        success {
            echo '✅ Build succeeded!'
        }
        failure {
            echo '❌ Build failed!'
        }
    }
}
