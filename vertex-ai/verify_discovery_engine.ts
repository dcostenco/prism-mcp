import { SearchServiceClient } from '@google-cloud/discoveryengine';

/**
 * Verifies if the Vertex AI Search (Discovery Engine) index is ready for queries.
 */
async function verifyIndex() {
  const projectId = process.env.DISCOVERY_ENGINE_PROJECT_ID || process.env.GCP_PROJECT_ID || '<your-gcp-project>';
  const location = process.env.DISCOVERY_ENGINE_LOCATION || 'global';
  const collectionId = process.env.DISCOVERY_ENGINE_COLLECTION || 'default_collection';
  const engineId = process.env.DISCOVERY_ENGINE_ENGINE_ID || '<your-engine-id>';
  const servingConfigId = process.env.DISCOVERY_ENGINE_SERVING_CONFIG || 'default_serving_config';

  // Pass projectId so the gRPC client sends the x-goog-user-project header
  // (required for Discovery Engine when using ADC / user credentials)
  const client = new SearchServiceClient({ projectId });

  // Construct the serving config path
  const servingConfig = `projects/${projectId}/locations/${location}/collections/${collectionId}/engines/${engineId}/servingConfigs/${servingConfigId}`;

  console.log(`🔍 Checking index status for: ${engineId}...`);
  console.log(`📍 Path: ${servingConfig}`);

  try {
    const request = {
      servingConfig,
      query: 'deep learning',
      pageSize: 5,
    };

    // Use rawResponse to get the full response object
    const [results, , rawResponse] = await client.search(request, { autoPaginate: false });

    // The response can be either the raw response (3rd element) or the results array (1st element)
    const response = rawResponse || {};
    const resultsList = Array.isArray(results) ? results : (response as any).results || [];

    if (resultsList.length > 0) {
      console.log('✅ Index is ACTIVE and returning results!');
      console.log(`📄 Results found: ${(response as any).totalSize || resultsList.length}`);
      
      resultsList.slice(0, 5).forEach((result: any, index: number) => {
        const doc = result.document || result;
        const data = doc.derivedStructData;
        // Protobuf Struct: access via .fields['key'].stringValue
        const title = data?.fields?.title?.stringValue
          || data?.fields?.htmlTitle?.stringValue
          || data?.title
          || doc?.name || 'No Title';
        const link = data?.fields?.link?.stringValue
          || data?.fields?.formattedUrl?.stringValue
          || data?.link
          || 'No Link';
        console.log(`\n[${index + 1}] ${title}`);
        console.log(`🔗 URL: ${link}`);
      });
    } else {
      console.log('⚠️ Index is active but returned 0 results. It might still be crawling.');
      console.log('Debug — raw response keys:', Object.keys(response));
      console.log('Debug — results type:', typeof results, Array.isArray(results) ? `(length: ${(results as any).length})` : '');
    }
  } catch (error: any) {
    if (error.message?.includes('not found') || error.message?.includes('indexer is not ready')) {
      console.log('⏳ Index is still being built or crawled. This can take 30-60 minutes.');
      console.log(`Original Error: ${error.message}`);
    } else {
      console.error('❌ Error querying Discovery Engine:', error.message);
      if (error.details) console.error('Details:', error.details);
    }
  }
}

verifyIndex();
