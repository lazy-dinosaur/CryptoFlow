/**
 * PM2 Ecosystem Configuration
 *
 * Usage:
 *   pm2 start ecosystem.config.js              # Start all services
 *   pm2 start ecosystem.config.js --only api   # Start API only
 *   pm2 restart api                            # Restart API (collector keeps running!)
 *   pm2 restart ml                             # Restart ML only
 *   pm2 logs                                   # View all logs
 *   pm2 logs collector                         # View collector logs
 *   pm2 status                                 # Check status
 */

module.exports = {
    apps: [
        {
            // Data Collector - runs 24/7, rarely restart
            name: 'collector',
            script: 'collector-standalone.js',
            cwd: './server',
            instances: 1,
            autorestart: true,
            watch: false,
            max_memory_restart: '300M',
            env: {
                NODE_ENV: 'production'
            },
            log_date_format: 'YYYY-MM-DD HH:mm:ss',
            error_file: './logs/collector-error.log',
            out_file: './logs/collector-out.log',
            merge_logs: true
        },
        {
            // API Server - can restart for updates
            name: 'api',
            script: 'api.js',
            cwd: './server',
            instances: 1,
            autorestart: true,
            watch: false,
            max_memory_restart: '300M',
            env: {
                NODE_ENV: 'production',
                PORT: 3000
            },
            log_date_format: 'YYYY-MM-DD HH:mm:ss',
            error_file: './logs/api-error.log',
            out_file: './logs/api-out.log',
            merge_logs: true
        },
        {
            // ML Service - can restart for updates
            name: 'ml',
            script: 'ml_service.py',
            cwd: './server',
            interpreter: 'python3',
            instances: 1,
            autorestart: true,
            watch: false,
            max_memory_restart: '400M',
            env: {
                NODE_ENV: 'production'
            },
            log_date_format: 'YYYY-MM-DD HH:mm:ss',
            error_file: './logs/ml-error.log',
            out_file: './logs/ml-out.log',
            merge_logs: true
        },
        {
            // ML Paper Trading - 3 strategies comparison
            name: 'ml-paper',
            script: 'ml_paper_trading.py',
            cwd: './server',
            interpreter: 'python3',
            interpreter_args: '-u',
            instances: 1,
            autorestart: true,
            watch: false,
            max_memory_restart: '500M',
            env: {
                NODE_ENV: 'production',
                PYTHONUNBUFFERED: '1'
            },
            log_date_format: 'YYYY-MM-DD HH:mm:ss',
            error_file: './logs/ml-paper-error.log',
            out_file: './logs/ml-paper-out.log',
            merge_logs: true
        }
    ]
};
