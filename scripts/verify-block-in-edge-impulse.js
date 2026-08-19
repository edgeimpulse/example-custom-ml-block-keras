const EdgeImpulseApi = require('edge-impulse-api').EdgeImpulseApi;
const spawn = require('child_process').spawn;
const Path = require('node:path');
const fs = require('node:fs');
const program = require('commander');

const packageVersion = JSON.parse(fs.readFileSync(Path.join(__dirname, 'package.json'), 'utf-8')).version;
const PROJECT_CONFIG_FILE = Path.join(__dirname, '..', '.ei-project-config.json');
const EI_BLOCK_CONFIG = Path.join(__dirname, '..', '.ei-block-config');

if (!fs.existsSync(PROJECT_CONFIG_FILE)) {
    console.log(`Missing ${PROJECT_CONFIG_FILE}. Run 'node scripts/configure-project.js' first.`);
    process.exit(1);
}
if (!fs.existsSync(EI_BLOCK_CONFIG)) {
    console.log(`Missing ${EI_BLOCK_CONFIG}`);
    process.exit(1);
}

const PROJECT_CONFIG = JSON.parse(fs.readFileSync(PROJECT_CONFIG_FILE, 'utf-8'));
if (!PROJECT_CONFIG.apiKey || !PROJECT_CONFIG.apiKey.startsWith('ei_')) {
    console.log(`API key (in .ei-project-config.json) does not start with "ei_"`);
    process.exit(1);
}
if (!Number.isInteger(PROJECT_CONFIG.impulseId)) {
    console.log(`Missing or invalid impulseId in .ei-project-config.json`);
    process.exit(1);
}
if (!Number.isInteger(PROJECT_CONFIG.learnBlockId)) {
    console.log(`Missing or invalid learnBlockId in .ei-project-config.json`);
    process.exit(1);
}

program
    .description('Verify blocks in Edge Impulse')
    .version(packageVersion)
    .option('--push-block', 'Push the block to Edge Impulse')
    .allowUnknownOption(false)
    .parse(process.argv);

// eslint-disable-next-line @typescript-eslint/no-floating-promises
(async () => {
    try {
        const pushBlock = !!program.pushBlock;

        const blockConfig = JSON.parse(await fs.promises.readFile(EI_BLOCK_CONFIG, 'utf-8'));
        if (blockConfig.version !== 2) {
            throw new Error(`${EI_BLOCK_CONFIG}, version is not "2" but "${blockConfig.version}"`);
        }
        if (!(blockConfig.config || { })['edgeimpulse.com']) {
            throw new Error(`${EI_BLOCK_CONFIG}, missing "config[edgeimpulse.com]". Did you run 'edge-impulse-blocks init'`);
        }
        if (!blockConfig.config['edgeimpulse.com']['organizationId']) {
            console.log(blockConfig.config['edgeimpulse.com']);
            throw new Error(`${EI_BLOCK_CONFIG}, missing "config[edgeimpulse.com][organizationId]". Did you run 'edge-impulse-blocks init'`);
        }
        if (!blockConfig.config['edgeimpulse.com']['id']) {
            throw new Error(`${EI_BLOCK_CONFIG}, missing "config[edgeimpulse.com][id]". Did you run 'edge-impulse-blocks push'`);
        }

        if (pushBlock) {
            console.log('Pushing block...')
            await spawnHelper('edge-impulse-blocks', ['push'], { cwd: Path.join(__dirname, '..') });
            console.log('Pushing block OK');
            console.log('');
        }

        const api = new EdgeImpulseApi();
        await api.authenticate({
            method: 'apiKey',
            apiKey: PROJECT_CONFIG.apiKey,
        });

        // list all projects (if you authenticate via API you just get one, the project for which you have the API key)
        let project = (await api.projects.listProjects()).projects[0];
        console.log('Project:', project.owner, '/', project.name);

        // select impulse / learn block
        let impulseRes = await api.impulse.getAllImpulses(project.id);

        const impulse = impulseRes.impulses.find(x => x.id === PROJECT_CONFIG.impulseId);
        if (!impulse) {
            throw new Error(`Could not find impulse with ID ${PROJECT_CONFIG.impulseId} (from .ei-project-config.json)`);
        }

        const learnBlock = impulse.learnBlocks.find(x => x.id === PROJECT_CONFIG.learnBlockId);
        if (!learnBlock) {
            throw new Error(`Could not find learn block with ID ${PROJECT_CONFIG.learnBlockId} (from .ei-project-config.json)`);
        }

        // and retrain with same config
        let trainJob = await api.jobs.trainKerasJob(project.id, learnBlock.id, {
            "trainTestSplit": 0.2,
            "customValidationMetadataKey": "",
            "autoClassWeights": false,
            "profileInt8": true,
            "mode": "visual",
            "visualLayers": [{
                "type": "transfer_organization",
                "organizationModelId": blockConfig.config['edgeimpulse.com']['id'],
            }],
            "augmentationPolicyImage": "none",
            "useLearnedOptimizer": false,
            "blockParameters": {},
            "customParameters": {   // <-- maps to parameters.json
                "epochs": "30",
                "learning-rate": "0.001",
            },
        });
        console.log('Created train job with ID', trainJob.id);

        await api.runJobUntilCompletion({
            type: 'project',
            projectId: project.id,
            jobId: trainJob.id,
        }, data => {
            process.stdout.write(data);
        });

        console.log('Train job completed');
    }
    catch (ex) {
        console.log('Failed to make a request', ex);
        process.exit(1);
    }
})();


function spawnHelper(
    command,
    args,
    opts
) {
    return new Promise((resolve, reject) => {
        const p = spawn(command, args, { env: process.env, cwd: opts.cwd });

        const allData = [];

        p.stdout.on('data', (data) => {
            process.stdout.write(data);
            allData.push(data);
            if (
                opts.readySignal &&
                data.toString().includes(opts.readySignal)
            ) {
                resolve(Buffer.concat(allData).toString('utf-8'));
            }
        });

        p.stderr.on('data', (data) => {
            process.stderr.write(data);
            allData.push(data);
        });

        p.on('error', () => {
            reject(
                'Error running command: ' +
                    Buffer.concat(allData).toString('utf-8')
            );
        });

        p.on('close', (code) => {
            if (code === 0 || opts.ignoreErrors === true) {
                resolve(Buffer.concat(allData).toString('utf-8'));
            } else {
                reject(
                    'Error code was not 0: ' +
                        Buffer.concat(allData).toString('utf-8')
                );
            }
        });
    });
}
