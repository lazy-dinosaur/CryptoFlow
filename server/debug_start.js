const fs = require('fs');

const logFile = 'DIAGNOSIS.txt';

function log(msg) {
    const time = new Date().toISOString();
    const line = `[${time}] ${msg}\n`;
    console.log(msg);
    fs.appendFileSync(logFile, line);
}

// Clear previous log
fs.writeFileSync(logFile, '--- STARTING DIAGNOSIS ---\n');

try {
    log('1. Checking Node version...');
    log(`Node: ${process.version}`);

    log('2. Loading Environment...');
    log(`PORT: ${process.env.PORT || 'Not Set'}`);

    log('3. Loading db.js...');
    try {
        const db = require('./db.js');
        log('✅ db.js loaded successfully.');
    } catch (e) {
        log('❌ CRASH loading db.js:');
        log(e.stack || e.toString());
        process.exit(1);
    }

    log('4. Loading collector.js...');
    try {
        require('./collector.js');
        log('✅ collector.js loaded successfully.');
    } catch (e) {
        log('❌ CRASH loading collector.js:');
        log(e.stack || e.toString());
        process.exit(1);
    }

    log('5. Loading api.js dependencies...');
    try {
        require('express');
        require('cors');
        require('ws');
        log('✅ express/cors/ws loaded successfully.');
    } catch (e) {
        log('❌ CRASH loading dependencies:');
        log(e.stack || e.toString());
        process.exit(1);
    }

    log('✅ ALL CHECKS PASSED. The issue is likely runtime, not load-time.');

} catch (e) {
    log('❌ UNEXPECTED TOP-LEVEL ERROR:');
    log(e.stack || e.toString());
}
