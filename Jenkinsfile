pipeline {
    agent any

    environment {
        // Explicit paths for Python and pip executables
        PYTHON = 'C:\\Users\\admin\\AppData\\Local\\Programs\\Python\\Python313\\python.exe'
        PIP    = 'C:\\Users\\admin\\AppData\\Local\\Programs\\Python\\Python313\\Scripts\\pip.exe'
        
    }

    stages {
        stage('Clone Repo') {
            steps {
                echo '📥 Cloning Customer Churn Prediction repository...'
                git branch: 'main', url: 'https://github.com/Ishaan-afk70/Customer-Churn-Prediction.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                echo '📦 Installing Python packages...'
                bat '''
                    %PIP% install --upgrade pip
                    %PIP% install -r requirements.txt
                    // Additional test dependencies
                    %PIP% install werkzeug flask pytest pytest-flask
                    pip list
                '''
            }
        }

        stage('Run Model Script') {
            steps {
                echo '🚀 Executing churn prediction script...'
                bat '%PYTHON% churn.py'
            }
        }

        stage('Run Tests') {
            steps {
                echo '🧪 Running unit tests...'
                bat '%PYTHON% -m pytest tests --maxfail=1 --disable-warnings -q'
            }
        }

        stage('Deploy (optional)') {
            steps {
                echo '🚢 Deployment logic placeholder.'
            }
        }
    }

    post {
        success {
            echo '✅ Pipeline completed successfully!'
        }
        failure {
            echo '❌ Pipeline failed. Review console output for details.'
        }
    }
}
